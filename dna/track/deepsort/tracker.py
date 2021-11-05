# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import List, Union, Tuple
import enum

import logging

from numpy.linalg import det

from dna.track.deepsort.detection import Detection
from .utils import find_overlaps, overlap_ratio, project
import dna
from dna import get_logger
import numpy as np
from dna.types import Box, Size2d
import kalman_filter
import linear_assignment
import iou_matching
from track import Track

_logger = get_logger("dna.track.deep_sort")
_HOT_CLOSE_THRESHOLD = 21
_OVERLAP_RATIO_THRESHOLD = 0.8
_MIN_OVERLAP_RATIO = 0.75

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

    def __init__(self, domain, metric, max_iou_distance=0.7, max_age=40, n_init=3, min_size=Size2d(25, 25), blind_regions=[]):
        self.domain = domain
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.min_size = min_size
        self.blind_regions = blind_regions
        self.new_track_iou_threshold = 0.55

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
        for track_idx, detection_idx in matches:
            track = self.tracks[track_idx]
            track.update(self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            # kwlee
            # track 영역이 image 전체의 영역에서 1/4 이상 벗어난 경우에는
            # 더 이상 추적하지 않는다.
            track = self.tracks[track_idx]
            bbox = Box.from_tlbr(track.to_tlbr())
            if bbox.is_valid():
                if overlap_ratio(bbox, self.domain) < 3/4:
                    track.mark_deleted()
                else:
                    track.mark_missed()
            else:
                track.mark_deleted()

        # kwlee
        # unmatched detection 중에서 다른 detection과 일정부분 이상 겹치는 경우에는
        # 새로운 track으로 간주되지 않게하기 위해 제거한다.
        if len(unmatched_detections) > 0 and len(detections) > 1:
            det_boxes = [Box.from_tlbr(d.to_tlbr()) for d in detections]

            def has_better_overlap(didx):
                def is_better(ov):
                    return ov[0] != didx \
                            and (ov[0] not in unmatched_detections \
                                or detections[ov[0]].confidence > detections[didx].confidence)

                box = det_boxes[didx]
                ovs = list(filter(is_better, find_overlaps(box, det_boxes, self.new_track_iou_threshold)))
                if len(ovs) > 0:
                    _logger.debug((f"remove an unmatched detection that overlaps with better one: "
                                    f"removed={didx}, better={ovs[0][0]}, ratio={ovs[0][1]:.2f}, "
                                    f"frame={dna.DEBUG_FRAME_IDX}"))
                    return True
                else:
                    return False

            unmatched_detections = [didx for didx in unmatched_detections if not has_better_overlap(didx)]

        for detection_idx in unmatched_detections:
            track = self._initiate_track(detections[detection_idx])
            self.tracks.append(track)
            self._next_id += 1

        # kwlee
        # update된 위치가 blind_region 안에 있는 경우는 delete된 것으로 간주한다.
        for track in self.tracks:
            if track.is_tentative():
                i = self.n_init
            if not track.is_deleted():
                tbox = Box.from_tlbr(track.to_tlbr())
                if tbox.width() < self.min_size.width and tbox.height() < self.min_size.height:
                    track.mark_deleted()
                else:
                    for r in self.blind_regions:
                        if r.contains(tbox):
                            track.mark_deleted()

        delete_tracks = [t for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # kwlee (첫번째로 검출되지 마자 delete되는 경우 그냥 삭제함)
        delete_tracks = [t for t in delete_tracks if t.age > 1]

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

    def _remove_overlaps(self, targets, detections, target_indices, candidate_indices):
        candidate_boxes = [Box.from_tlbr(detections[didx].to_tlbr()) for didx in candidate_indices]

        overlaps = set()
        for tidx in target_indices:
            box = Box.from_tlbr(targets[tidx].to_tlbr())
            ovs = find_overlaps(box, candidate_boxes, _OVERLAP_RATIO_THRESHOLD)
            if len(ovs) > 0:
                ov_idxes = [candidate_indices[idx] for idx in project(ovs, 0)]
                overlaps.update(ov_idxes)
                if _logger.isEnabledFor(logging.DEBUG):
                    ov_str = ",".join([f"{candidate_indices[i]}({r:.2f})" for i, r in ovs])
                    _logger.debug((f"remove hot-track[{tidx}]'s overlaps: "
                                    f"track={targets[tidx]}, det={ov_str}, frame={dna.DEBUG_FRAME_IDX}"))
        non_overlaps = set(candidate_indices) - overlaps
        return list(non_overlaps)
        
    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.q
        hot_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() and t.time_since_update <= 1]
        temp_lost_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() and t.time_since_update > 1]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        dist_cost = self.distance_cost(self.tracks, detections)
        if dna.DEBUG_PRINT_COST:
            self.print_dist_cost(dist_cost, 999)

        # STEP 1: active한 track에 독점적으로 가까운 detection이 존재하면, association시킨다.
        matches, unmatched_hot_tracks, unmatched_detections \
             = linear_assignment.matching_by_close_distance(dist_cost, _HOT_CLOSE_THRESHOLD,
                                                            self.tracks, detections, hot_tracks)
        unmatched_tracks = unmatched_hot_tracks + temp_lost_tracks

        # active track과 binding된 detection과 상당히 겹치는 detection들을 제거한다.
        if len(matches) > 0 and len(unmatched_detections) > 0:
            matched_dets = project(matches, 1)
            unmatched_detections = self._remove_overlaps(detections, detections,
                                                        matched_dets, unmatched_detections)

        if (len(unmatched_tracks) > 0 or len(unconfirmed_tracks) > 0) and len(unmatched_detections) > 0:
            metric_cost = self.metric_cost(self.tracks, detections)
            cmatrix = linear_assignment.combine_cost_matrices(metric_cost, dist_cost, self.tracks, detections)
            if dna.DEBUG_PRINT_COST:
                self.print_metrix_cost(metric_cost, unmatched_tracks+unconfirmed_tracks, unmatched_detections)
                self.print_metrix_cost(cmatrix, unmatched_tracks+unconfirmed_tracks, unmatched_detections)

            if len(unmatched_tracks) > 0:
                # STEP 2
                matches_1, unmatched_tracks, unmatched_detections =\
                    linear_assignment.matching_by_total_cost(cmatrix, unmatched_tracks, unmatched_detections)
                matches += matches_1

                # 만일 STEP 1에서 가까운 detection이 존재했지만, 해당 detection 독점적이지 아니어서 association되지
                # 못해 binding되지 못했지만, STEP 2 과정에서 경쟁하던 track이 다른 detection에 association되어
                # 이제는 독점적으로 된 경우를 처리한다.
                matched_hot_tracks = [m[0] for m in matches_1 if m[0] in unmatched_hot_tracks]
                unmatched_hot_tracks = [tidx for tidx in unmatched_hot_tracks if tidx not in matched_hot_tracks]
                if len(unmatched_hot_tracks) > 0 and len(unmatched_detections) > 0:
                    matches_h, unmatched_hot_tracks, unmatched_detections \
                        = linear_assignment.matching_by_close_distance(dist_cost, _HOT_CLOSE_THRESHOLD,
                                                                        self.tracks, detections,
                                                                        unmatched_hot_tracks, unmatched_detections)
                    matches += matches_h
                    for m in matches_h:
                        unmatched_tracks.remove(m[0])

            if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
                matches_0, unmatched_unconfirmed_tracks, unmatched_detections =\
                        linear_assignment.matching_by_close_distance(cmatrix, 0.2, self.tracks, detections,
                                                                        unconfirmed_tracks, unmatched_detections)
                matches += matches_0
            else:
                unmatched_unconfirmed_tracks = unconfirmed_tracks

            if len(unmatched_unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
                matches_2, unmatched_unconfirmed_tracks, unmatched_detections =\
                    linear_assignment.matching_by_total_cost(cmatrix, unmatched_unconfirmed_tracks, unmatched_detections)
                matches += matches_2
                unmatched_tracks += unmatched_unconfirmed_tracks
        else:
            unmatched_tracks += unconfirmed_tracks

        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            matches_o, unmatched_tracks, unmatched_detections \
                = linear_assignment.matching_by_overlap(_MIN_OVERLAP_RATIO, self.tracks, detections,
                                                        unmatched_tracks, unmatched_detections)
            matches += matches_o

        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            matches_2, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance,
                                                    self.tracks, detections,
                                                    unmatched_tracks, unmatched_detections)
            matches += matches_2

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
                # print(dist_matrix[row, :])
                # print(self.kf.gating_distance(track.mean, track.covariance, measurements, True))
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
    if v == 9.99:
        return "    "
    else:
        return f"{v:.2f}"
    ###############################################################################################################