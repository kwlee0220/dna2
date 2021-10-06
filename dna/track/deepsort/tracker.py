# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from dna.track.deepsort.detection import Detection
import numpy as np
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

    def __init__(self, metric, max_iou_distance=0.7, max_age=40, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

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
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            track = self._initiate_track(detections[detection_idx])
            self.tracks.append(track)
            self._next_id += 1

        delete_tracks = [t for t in self.tracks if t.is_deleted()]
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
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)

            # 각 track의 위치가 detection들 사이의 거리가 mahalanobis 거리를 기준으로 일정 거리 이상인 경우
            # 해당 cost_matrix 상의 거리 값을 무효화 (INFTY_COST) 한다.
            # (즉, track 위치와 detection과 거리가 너무 멀면 appearance가 비슷해도 무시한다.)
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, dets,
                                                            track_indices, detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks, unmatched_detections = \
            linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age,
                                                self.tracks, detections,
                                                confirmed_tracks)

        # Confirmed track들 중에서 appearance metric으로 assign되지 못한 track들에서 대해서만
        # IoU 거리를 통한 association을 시도한다.
        matches_b, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance,
                                                self.tracks, detections,
                                                unmatched_tracks, unmatched_detections)

        # Unconfirmed track들에 대해 IoU distance를 사용하여 association을 시도한다.
        # 근데, 왜 unconfirmed track들에 대해서 appearance를 사용하는 matching을 실시하지 않는
        # 이유를 잘 모르겠음.
        matches_c, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(iou_matching.iou_cost, self.max_iou_distance,
                                                self.tracks, detections,
                                                unconfirmed_tracks, unmatched_detections)

        return matches_a + matches_b + matches_c, unmatched_tracks_a + unmatched_tracks_b, unmatched_detections

    def _initiate_track(self, detection: Detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        return Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection)
