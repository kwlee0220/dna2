# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Union, Tuple
import enum

import logging

from numpy.linalg import det

from dna.track.deepsort.detection import Detection
from . import matcher
from .utils import all_indices, intersection, subtract, project, overlap_ratios, find_overlaps_threshold
import dna
from dna import get_logger
import numpy as np
from dna.types import Box, Size2d
import kalman_filter
import linear_assignment
import iou_matching
from track import Track

_logger = get_logger("dna.track.deep_sort")

_HOT_DIST_THRESHOLD = 21
_TOTAL_COST_THRESHOLD = 0.5
_TOTAL_COST_THRESHOLD_WEAK = 0.75
_TOTAL_COST_THRESHOLD_STRONG = 0.2
_REMOVE_OVERLAP_RATIO_THRESHOLD = 0.8
_OVERLAP_MATCH_HIGH = 0.75
_OVERLAP_MATCH_LOW = 0.55

_OBSOLTE_TRACK_SIZE = 5

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted. Default 30
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, domain, metric, max_iou_distance, max_age, n_init, min_size, blind_regions=[]):
        self.domain = domain
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.min_size = min_size
        self.blind_regions = blind_regions
        self.new_track_overlap_threshold = 0.75

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for tidx, didx in matches:
            self.tracks[tidx].update(self.kf, detections[didx])

        t_boxes = [Box.from_tlbr(track.to_tlbr()) for track in self.tracks]

        for tidx in unmatched_tracks:
            # track 영역이 image 전체의 영역에서 1/4 이상 벗어난 경우에는 더 이상 추적하지 않는다.
            track = self.tracks[tidx]
            ratios = overlap_ratios(t_boxes[tidx], self.domain)
            if ratios[0] < 0.85:
                track.mark_deleted()
            else:
                track.mark_missed()

        # Temporarily lost된 track의 bounding-box의 크기가 일정 이하이면 delete시킨다.
        # for tidx in range(len(self.tracks)):
        #     track = self.tracks[tidx]
        #     if track.is_confirmed() and track.time_since_update > 1:
        #         size = t_boxes[tidx].size().to_int()
        #         if size.width < _OBSOLTE_TRACK_SIZE or size.height < _OBSOLTE_TRACK_SIZE:
        #             _logger.debug((f"delete too small temp-lost track[{track.track_id}:{track.time_since_update}], "
        #                             f"size={size}, frame={dna.DEBUG_FRAME_IDX}"))
        #             track.mark_deleted()

        # track의 bounding-box가 blind_region에 포함된 경우는 delete시킨다.
        for tidx in range(len(self.tracks)):
            for r in self.blind_regions:
                if r.contains(t_boxes[tidx]):
                    self.tracks[tidx].mark_deleted()

        # confirmed track과 너무 가까운 tentative track들을 제거한다.
        # 일반적으로 이런 track들은 이전 frame에서 한 물체의 여러 detection 검출을 통해 track이 생성된
        # 경우가 많아서 이를 제거하기 위함이다.
        matcher.delete_overlapped_tentative_tracks(self.tracks, _REMOVE_OVERLAP_RATIO_THRESHOLD)

        # kwlee
        # unmatched detection 중에서 다른 detection과 일정부분 이상 겹치는 경우에는
        # 새로운 track으로 간주되지 않게하기 위해 제거한다.
        if len(unmatched_detections) > 0 and len(detections) > 1:
            det_boxes = [Box.from_tlbr(d.to_tlbr()) for d in detections]
            non_overlapped = unmatched_detections.copy()
            for didx in unmatched_detections:
                box = det_boxes[didx]

                # 일정 크기 이하의 detection들은 무시한다.
                if box.width() < self.min_size.width or box.height() < self.min_size.height:
                    non_overlapped.remove(didx)
                    continue

                confi = detections[didx].confidence
                for ov in find_overlaps_threshold(box, det_boxes, self.new_track_overlap_threshold):
                    if ov[0] != didx and (ov[0] not in unmatched_detections or detections[ov[0]].confidence > confi):
                        _logger.debug((f"remove an unmatched detection that overlaps with better one: "
                                        f"removed={didx}, better={ov[0]}, ratios={max(ov[1]):.2f}, "
                                        f"frame={dna.DEBUG_FRAME_IDX}"))
                        non_overlapped.remove(didx)
                        break
            unmatched_detections = non_overlapped

        for didx in unmatched_detections:
            track = self._initiate_track(detections[didx])
            self.tracks.append(track)
            self._next_id += 1

        delete_tracks = [t for t in self.tracks if t.is_deleted() and t.age > 1]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in confirmed_tracks:
            features += track.features
            targets += [track.track_id for _ in track.features]

            # # 왜 이전 feature들을 유지하지 않지?
            track.features = [track.features[-1]] #Retain most recent feature of the track.
            # track.features = track.features[-5:]

        active_targets = [t.track_id for t in confirmed_tracks]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

        return delete_tracks
        
    def _match(self, detections):
        if len(detections) == 0:
            return [], all_indices(self.tracks), detections

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        hot_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() and t.time_since_update <= 3]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        dist_cost = self.distance_cost(self.tracks, detections)
        if dna.DEBUG_PRINT_COST:
            self.print_dist_cost(dist_cost, 999)

        matches = []
        unmatched_tracks = all_indices(self.tracks)
        unmatched_detections = all_indices(detections)

        #####################################################################################################
        ################ Hot track에 한정해서 matching 실시
        #####################################################################################################

        # STEP 1: hot track에 독점적으로 가까운 detection이 존재하면, association시킨다.
        if len(detections) > 0 and len(hot_tracks) > 0:
            matches_hot, unmatched_hot, unmatched_detections \
                = matcher.matching_by_excl_best(dist_cost, _HOT_DIST_THRESHOLD, hot_tracks, unmatched_detections)
            matches += matches_hot
            unmatched_tracks = subtract(unmatched_tracks, project(matches_hot, 0))

            # active track과 binding된 detection과 상당히 겹치는 detection들을 제거한다.
            if len(matches_hot) > 0 and len(unmatched_detections) > 0:
                unmatched_detections = matcher.remove_overlaps(detections, detections, _REMOVE_OVERLAP_RATIO_THRESHOLD,
                                                                project(matches_hot, 1), unmatched_detections)
        else:
            unmatched_hot = hot_tracks

        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            metric_cost = self.metric_cost(self.tracks, detections)
            cmatrix = matcher.combine_cost_matrices(metric_cost, dist_cost, self.tracks, detections)
            if dna.DEBUG_PRINT_COST:
                self.print_metrix_cost(metric_cost, unmatched_tracks, unmatched_detections)
                self.print_metrix_cost(cmatrix, unmatched_tracks, unmatched_detections)

        if len(unmatched_hot) > 0 and len(unmatched_detections) > 0:
            # STEP 2
            matches_s, _, unmatched_detections =\
                matcher.matching_by_hungarian(cmatrix, _TOTAL_COST_THRESHOLD,
                                                unmatched_hot, unmatched_detections)
            matches += matches_s
            unmatched_tracks = subtract(unmatched_tracks, project(matches_s, 0))

        #####################################################################################################
        ################ Confirmed track에 한정해서 강한 threshold를 사용해서  matching 실시
        #####################################################################################################
        # confirmed track들 중에서 time_since_update가 큰 경우는 motion 정보의 variance가 큰 상태라
        # 실제로 먼 거리에 있는 detection과의 거리가 그리 멀지 않게되기 때문에 주의해야 함.
        unmatched_confirmed_tracks = subtract(confirmed_tracks, project(matches, 0))
        if len(unmatched_confirmed_tracks) > 0 and len(unmatched_detections) > 0:
            # STEP 2
            # confirmed track들 사이에서 tight한 threshold를 사용해서 matching을 실시한다.
            matches_s, unmatched_confirmed_tracks, unmatched_detections =\
                matcher.matching_by_hungarian(cmatrix, _TOTAL_COST_THRESHOLD_STRONG,
                                                unmatched_confirmed_tracks, unmatched_detections)
            matches += matches_s
            unmatched_tracks = subtract(unmatched_tracks, project(matches_s, 0))

        #####################################################################################################
        ################ Tentative track에 penality를 부여한 weighted matrix를 
        ################ 사용하여 전체 track에 대해 matching 실시
        #####################################################################################################
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            unconfirmed_weights = np.array([1 if track.is_confirmed() else 3 for track in self.tracks])
            weighted_matrix = np.multiply(cmatrix, unconfirmed_weights[:, np.newaxis])
            if dna.DEBUG_PRINT_COST:
                self.print_metrix_cost(weighted_matrix, unmatched_tracks, unmatched_detections)

            matches_s, unmatched_tracks, unmatched_detections =\
                    matcher.matching_by_hungarian(weighted_matrix, _TOTAL_COST_THRESHOLD,
                                                    unmatched_tracks, unmatched_detections)
            matches += matches_s
            
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            ovr_matrix = matcher.overlap_cost(self.tracks, detections, unmatched_tracks, unmatched_detections)

        #####################################################################################################
        ################ 겹침 정도로 gating하고 조금 더 느슨한 threshold를 사용하여 matching 실시.
        #####################################################################################################
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            weighted_matrix[ovr_matrix < 0.35] = 9.99
            if dna.DEBUG_PRINT_COST:
                self.print_metrix_cost(weighted_matrix, unmatched_tracks, unmatched_detections)
            matches_s, unmatched_tracks, unmatched_detections =\
                    matcher.matching_by_hungarian(weighted_matrix, _TOTAL_COST_THRESHOLD_WEAK,
                                                    unmatched_tracks, unmatched_detections)
            matches += matches_s

        # #####################################################################################################
        # ################ Confirmed track에 한정해서  matching 실시
        # #####################################################################################################

        # # confirmed track들 중에서 time_since_update가 큰 경우는 motion 정보의 variance가 큰 상태라
        # # 실제로 먼 거리에 있는 detection과의 거리가 그리 멀지 않게되기 때문에 주의해야 함.

        # unmatched_confirmed_tracks = unmatched_hot_tracks + tl_tracks
        # if len(unmatched_confirmed_tracks) > 0 and len(unmatched_detections) > 0:
        #     # STEP 2
        #     # confirmed track들 사이에서 total_cost를 사용해서 matching을 실시한다.
        #     matches_s, unmatched_confirmed_tracks, unmatched_detections =\
        #         matcher.matching_by_hungarian(cmatrix, _TOTAL_COST_THRESHOLD, unmatched_confirmed_tracks, unmatched_detections)
        #     matches += matches_s

        #     # STEP 3
        #     # 만일 STEP 1에서 가까운 detection이 존재했지만, 해당 detection이 타 track 때문에 독점적이지 아니어서 match되지 못했지만,
        #     # STEP 2 과정에서 경쟁하던 track이 다른 detection에 match되어 이제는 독점적으로 된 경우를 처리한다.
        #     unmatched_hot_tracks = intersection(unmatched_confirmed_tracks, hot_tracks)
        #     if len(unmatched_hot_tracks) > 0 and len(matches_s) > 0 and len(unmatched_detections) > 0:
        #         matches_s, unmatched_hot_tracks, unmatched_detections \
        #             = matcher.matching_by_excl_best(dist_cost, _HOT_DIST_THRESHOLD, unmatched_hot_tracks, unmatched_detections)
        #         matches += matches_s
        #         unmatched_confirmed_tracks = subtract(unmatched_confirmed_tracks, project(matches_s, 1))

        #####################################################################################################
        ################ Tentative track에 한정해서 matching 실시
        #####################################################################################################

        unmatched_unconfirmed_tracks = intersection(unmatched_tracks, unconfirmed_tracks)
        # if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
        #     matches_s, unmatched_unconfirmed_tracks, unmatched_detections =\
        #             matcher.matching_by_excl_best(cmatrix, _TOTAL_COST_HIGH_THRESHOLD, unmatched_unconfirmed_tracks, unmatched_detections)
        #     total_matches += matches_s

        if len(unmatched_unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
            matches_s, unmatched_unconfirmed_tracks, unmatched_detections =\
                matcher.matching_by_hungarian(cmatrix, _TOTAL_COST_THRESHOLD, unmatched_unconfirmed_tracks, unmatched_detections)
            matches += matches_s
            unmatched_tracks = subtract(unmatched_tracks, project(matches_s, 0))

        # unmatched_tracks = unmatched_confirmed_tracks + unmatched_unconfirmed_tracks

        #####################################################################################################
        ################ 남은 unmatched track에 대해서 matching 실시
        ################ 이때, 너무 넉넉하게 matching하면 new track으로 될 detection들이 억지로 기존 track에 matching되는 경우 있음
        #####################################################################################################

        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            matches_s, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance,
                                                    self.tracks, detections,
                                                    unmatched_tracks, unmatched_detections)
            matches += matches_s

        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            def match_overlap(r1, r2, iou):
                return max(r1, r2) >= _OVERLAP_MATCH_HIGH and min(r1, r2) >= _OVERLAP_MATCH_LOW

            matches_s, unmatched_tracks, unmatched_detections \
                = matcher.matching_by_overlap(self.tracks, detections, match_overlap, unmatched_tracks, unmatched_detections)
            matches += matches_s

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection: Detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        return Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection)

    ###############################################################################################################
    # kwlee
    def metric_cost(self, tracks, detections):
        features = np.array([det.feature for det in detections])
        targets = np.array([track.track_id for track in tracks])
        return self.metric.distance(features, targets)

    # kwlee
    def distance_cost(self, tracks, detections, only_position=False):
        dist_matrix = np.zeros((len(tracks), len(detections)))
        if len(tracks) > 0 and len(detections) > 0:
            measurements = np.asarray([det.to_xyah() for det in detections])
            for row, track in enumerate(tracks):
                dist_matrix[row, :] = self.kf.gating_distance(track.mean, track.covariance,
                                                                measurements, only_position)
        return dist_matrix

    # kwlee
    def print_cost(self, metric_cost, dist_cost):
        dist_cost = dist_cost.copy()
        dist_cost[dist_cost > 999] = 999

        for tidx, track in enumerate(self.tracks):
            dists = [int(round(v)) for v in dist_cost[tidx]]
            metrics = [round(v, 2) for v in metric_cost[tidx]]
            track_str = f"[{tidx:02d}]{track.track_id:03d}({track.state},{track.time_since_update})"
            cost_str = ', '.join([f"({v1:3d}, {v2:.2f})" for v1, v2 in zip(dists, metrics)])
            print(f"{track_str}: {cost_str}")

    def print_dist_cost(self, dist_cost, trim_overflow=None):
        if trim_overflow:
            dist_cost = dist_cost.copy()
            dist_cost[dist_cost > trim_overflow] = trim_overflow

        for tidx, track in enumerate(self.tracks):
            dists = [int(round(v)) for v in dist_cost[tidx]]
            track_str = f"{tidx:02d}: {track.track_id:03d}({track.state},{track.time_since_update:02d})"
            dist_str = ', '.join([f"{v:3d}" if v != trim_overflow else "   " for v in dists])
            print(f"{track_str}: {dist_str}")

    def print_metrix_cost(self, metric_cost, task_indices=None, detection_indices=None):
        if not task_indices:
            task_indices = list(range(len(self.tracks)))
        if not detection_indices:
            detection_indices = list(range(metric_cost.shape[1]))

        col_exprs = []
        for c in range(metric_cost.shape[1]):
            if c in detection_indices:
                col_exprs.append(f"{c:-5d}")
            else:
                col_exprs.append("-----")
        print("              ", ",".join(col_exprs))

        for tidx, track in enumerate(self.tracks):
            costs = [round(v, 2) for v in metric_cost[tidx]]
            track_str = f"{tidx:02d}: {track.track_id:03d}({track.state},{track.time_since_update:02d})"
            # dist_str = ', '.join([f"{v:.2f}" for v in costs])
            dist_str = ', '.join([_pattern(i,v) for i, v in enumerate(costs)])
            tag = '*' if tidx in task_indices else ' '
            print(f"{tag}{track_str}: {dist_str}")

def _pattern(i,v):
    if v >= 9.99:
        return "    "
    else:
        return f"{v:.2f}"
    ###############################################################################################################