# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from dna.track.deepsort.detection import Detection
import dna
import numpy as np
from dna.types import Box
import kalman_filter
import linear_assignment
import iou_matching
from track import Track

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

    def __init__(self, domain, metric, max_iou_distance=0.7, max_age=40, n_init=3, blind_regions=[]):
        self.domain = domain
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.blind_regions = blind_regions

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
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            track = self.tracks[track_idx]
            track.update(self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            # kwlee
            # track 영역이 image 전체의 영역에서 1/3 이상 벗어난 경우에는
            # 더 이상 추적하지 않는다.
            track = self.tracks[track_idx]
            bbox = Box.from_tlbr(track.to_tlbr())
            if bbox.is_valid():
                intersection = self.domain.intersection(bbox)
                inter_area = intersection.area() if intersection else 0
                if (inter_area / bbox.area()) < 2/3:
                    track.mark_deleted()
                else:
                    track.mark_missed()
            else:
                track.mark_deleted()

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
        
    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.q
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # ##############################################################################################
        # # kwlee
        dist_cost = self.distance_cost(self.tracks, detections)
        if dna.DEBUG_PRINT_COST:
            self.print_dist_cost(dist_cost, 999)
        matches, unmatched_tracks, unmatched_detections \
             = linear_assignment.matching_by_distance(dist_cost, self.tracks, detections, confirmed_tracks)

        unmatched_tracks += unconfirmed_tracks
        if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            metric_cost = self.metric_cost(self.tracks, detections)
            cmatrix = linear_assignment.combine_cost_matrices(metric_cost, dist_cost, self.tracks, detections)
            if dna.DEBUG_PRINT_COST:
                self.print_metrix_cost(metric_cost)
                print("-----------------------------------")
                self.print_metrix_cost(cmatrix)
            matches_1, unmatched_tracks, unmatched_detections =\
                    linear_assignment.matching_by_total_cost(cmatrix, self.tracks, detections,
                                                            unmatched_tracks, unmatched_detections)
            matches += matches_1

        if len(unmatched_tracks) > 0:
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
    import math
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
            dist_str = ', '.join([f"{v:3d}" for v in dists])
            print(f"{track_str}: {dist_str}")

    def print_metrix_cost(self, metric_cost):
        for tidx, track in enumerate(self.tracks):
            costs = [round(v, 2) for v in metric_cost[tidx]]
            track_str = f"{tidx:02d}: {track.track_id:03d}({track.state},{track.time_since_update:02d})"
            dist_str = ', '.join([f"{v:.2f}" for v in costs])
            print(f"{track_str}: {dist_str}")
    ###############################################################################################################