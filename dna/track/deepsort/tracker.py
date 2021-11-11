# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Union, Tuple
import enum

import logging

from numpy.linalg import det

from dna.track.deepsort.detection import Detection
from . import matcher, utils
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
_COST_THRESHOLD = 0.5
_COST_THRESHOLD_WEAK = 0.75
_COST_THRESHOLD_STRONG = 0.2
_REMOVE_OVERLAP_RATIO_THRESHOLD = 0.75
_OVERLAP_MATCH_HIGH = 0.75
_OVERLAP_MATCH_LOW = 0.55

_OBSOLTE_TRACK_SIZE = 5

class Tracker:
    def __init__(self, domain, metric, params):
        self.domain = domain
        self.metric = metric
        self.params = params
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

        t_boxes = [utils.track_to_box(track) for track in self.tracks]

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

        # track의 bounding-box가 exit_region에 포함된 경우는 delete시킨다.
        for tidx in range(len(self.tracks)):
            tbox = t_boxes[tidx]
            if any(r.contains(t_boxes[tidx]) for r in self.params.exit_zones):
                self.tracks[tidx].mark_deleted()
            elif any(r.contains(t_boxes[tidx]) for r in self.params.blind_zones):
                self.tracks[tidx].mark_deleted()

        # confirmed track과 너무 가까운 tentative track들을 제거한다.
        # 일반적으로 이런 track들은 이전 frame에서 한 물체의 여러 detection 검출을 통해 track이 생성된
        # 경우가 많아서 이를 제거하기 위함이다.
        matcher.delete_overlapped_tentative_tracks(self.tracks, _REMOVE_OVERLAP_RATIO_THRESHOLD)

        # kwlee
        # unmatched detection 중에서 다른 detection과 일정부분 이상 겹치는 경우에는
        # 새로운 track으로 간주되지 않게하기 위해 제거한다.
        if len(unmatched_detections) > 0:
            det_boxes = [Box.from_tlbr(d.to_tlbr()) for d in detections]
            non_overlapped = unmatched_detections.copy()
            for didx in unmatched_detections:
                box = det_boxes[didx]

                # 일정 크기 이하의 detection들은 무시한다.
                if box.width() < self.params.min_size.width or box.height() < self.params.min_size.height:
                    non_overlapped.remove(didx)
                    continue

                # Exit 영역에 포함되는 detection들은 무시한다
                if any(region.contains(box) for region in self.params.exit_zones):
                    non_overlapped.remove(didx)
                    # _logger.debug((f"remove an unmatched detection contained in a blind region: "
                    #                 f"removed={didx}, frame={dna.DEBUG_FRAME_IDX}"))
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
            if dna.DEBUG_PRINT_COST:
                print("[hot, only_dist]:", self.matches_str(matches_hot))
            unmatched_tracks = subtract(unmatched_tracks, project(matches_hot, 0))

            # active track과 binding된 detection과 상당히 겹치는 detection들을 제거한다.
            if len(matches_hot) > 0 and len(unmatched_detections) > 0:
                unmatched_detections = matcher.remove_overlaps(detections, detections, _REMOVE_OVERLAP_RATIO_THRESHOLD,
                                                                project(matches_hot, 1), unmatched_detections)
        else:
            unmatched_hot = hot_tracks

        #####################################################################################################
        ########## 통합 비용 행렬을 생성한다.
        ########## cmatrix: 통합 비용 행렬
        ########## ua_matrix: unconfirmed track을 고려한 통합 비용 행렬
        #####################################################################################################
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            metric_cost = self.metric_cost(self.tracks, detections)
            cmatrix, cmask = matcher.combine_cost_matrices(metric_cost, dist_cost, self.tracks, detections)
            cmatrix[cmask] = 9.99
            ua_matrix = cmatrix
        if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
            hot_mask = matcher.hot_unconfirmed_mask(cmatrix, 0.1, unconfirmed_tracks, unmatched_detections)
            ua_matrix = matcher.create_matrix(cmatrix, 9.99, hot_mask)

        if len(unmatched_hot) > 0 and len(unmatched_detections) > 0:
            matrix = matcher.create_matrix(ua_matrix, _COST_THRESHOLD_STRONG)
            if dna.DEBUG_PRINT_COST:
                self.print_matrix(matrix, _COST_THRESHOLD_STRONG, unmatched_hot, unmatched_detections)

            matches_s, _, unmatched_detections =\
                matcher.matching_by_hungarian(matrix, _COST_THRESHOLD_STRONG, unmatched_hot, unmatched_detections)
            if dna.DEBUG_PRINT_COST:
                print("[hot, combined]:", self.matches_str(matches_s))
            matches += matches_s
            unmatched_tracks = subtract(unmatched_tracks, project(matches_s, 0))
        else:
            matrix = None

        #####################################################################################################
        ################ Tentative track에 한정해서 강한 threshold를 사용해서  matching 실시
        #####################################################################################################
        if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
            matrix = matcher.create_matrix(ua_matrix, _COST_THRESHOLD_STRONG) if matrix is None else matrix
            matches_s, _, unmatched_detections =\
                matcher.matching_by_hungarian(matrix, _COST_THRESHOLD_STRONG, unconfirmed_tracks, unmatched_detections)
            if dna.DEBUG_PRINT_COST:
                print("[tentative, combined]:", self.matches_str(matches_s))
            matches += matches_s
            unmatched_tracks = subtract(unmatched_tracks, project(matches_s, 0))

        #####################################################################################################
        ################ 전체 track에 대해 matching 실시
        ################ Tentative track에게 약간의 penalty를 부여함
        #####################################################################################################
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            unconfirmed_weights = np.array([1 if track.is_confirmed() else 2 for track in self.tracks])
            weighted_matrix = np.multiply(cmatrix, unconfirmed_weights[:, np.newaxis])
            matrix = matcher.create_matrix(weighted_matrix, _COST_THRESHOLD)
            if dna.DEBUG_PRINT_COST:
                self.print_matrix(matrix, _COST_THRESHOLD, unmatched_tracks, unmatched_detections)

            matches_s, unmatched_tracks, unmatched_detections =\
                matcher.matching_by_hungarian(matrix, _COST_THRESHOLD, unmatched_tracks, unmatched_detections)
            if dna.DEBUG_PRINT_COST:
                print("[all, combined]:", self.matches_str(matches_s))
            matches += matches_s

        #####################################################################################################
        ################ 겹침 정도로 gating하고 조금 더 느슨한 threshold를 사용하여 matching 실시.
        #####################################################################################################
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            iou_matrix = matcher.iou_matrix(self.tracks, detections, unmatched_tracks, unmatched_detections)
            matrix = matcher.create_matrix(weighted_matrix, _COST_THRESHOLD_WEAK)
            matrix[iou_matrix < 0.1] = _COST_THRESHOLD_WEAK + 0.00001
            if dna.DEBUG_PRINT_COST:
                self.print_matrix(matrix, _COST_THRESHOLD_WEAK, unmatched_tracks, unmatched_detections)

            matches_s, unmatched_tracks, unmatched_detections =\
                matcher.matching_by_hungarian(matrix, _COST_THRESHOLD_WEAK, unmatched_tracks, unmatched_detections)
            if dna.DEBUG_PRINT_COST:
                print("[all, gated_weak]:", self.matches_str(matches_s))
            matches += matches_s

        #####################################################################################################
        ################ 남은 unmatched track에 대해서 겹침 정보를 기반으로 matching 실시
        #####################################################################################################
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            matches_s, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching(iou_matching.iou_cost, self.params.max_iou_distance,
                                                    self.tracks, detections,
                                                    unmatched_tracks, unmatched_detections)
            matches += matches_s

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection: Detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        return Track(mean, covariance, self._next_id, self.params.n_init, self.params.max_age, detection)

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

    def matches_str(self, matches):
        return ",".join([f"({self.tracks[tidx].track_id}, {didx})" for tidx, didx in matches])

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

    def print_matrix(self, matrix, threshold, task_indices=None, detection_indices=None):
        def pattern(v):
            return "    " if v > threshold else f"{v:.2f}"

        if not task_indices:
            task_indices = list(range(len(self.tracks)))
        if not detection_indices:
            detection_indices = list(range(matrix.shape[1]))

        col_exprs = []
        for c in range(matrix.shape[1]):
            if c in detection_indices:
                col_exprs.append(f"{c:-5d}")
            else:
                col_exprs.append("-----")
        print("              ", ",".join(col_exprs))

        for tidx, track in enumerate(self.tracks):
            track_str = f"{tidx:02d}: {track.track_id:03d}({track.state},{track.time_since_update:02d})"
            dist_str = ', '.join([pattern(v) for v in matrix[tidx]])
            tag = '*' if tidx in task_indices else ' '
            print(f"{tag}{track_str}: {dist_str}")

    ###############################################################################################################