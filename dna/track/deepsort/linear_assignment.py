# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

import kalman_filter
import dna
import dna.utils as du


INFTY_COST = 1e+5


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None,
                        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_sum_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections

# # kwlee
# def find_track_info_from_matches(matches, tracks, target_ids, track_indices, det_indices, matrix):
#     for m in matches:
#         track_idx = m[0]
#         track = tracks[track_idx]
#         if track.track_id in target_ids:
#             row_idx = track_indices.index(track_idx)
#             row = matrix[row_idx]
#             return track, reorder_matrix_row(row, det_indices)
#     return None, None

# # kwlee
# def find_from_unmatched_tracks(unmatched_tracks, tracks, target_ids, track_indices, det_indices, matrix):
#     for track_idx in unmatched_tracks:
#         track = tracks[track_idx]
#         if track.track_id in target_ids:
#             row_idx = track_indices.index(track_idx)
#             row = matrix[row_idx]
#             return track, reorder_matrix_row(row, det_indices)
#     return None, None

# # kwlee
# def reorder_matrix_row(row, detection_indices):
#     pairs = sorted([(idx, v) for idx, v in enumerate(detection_indices)], key=lambda t: t[1])
#     ordereds = [row[pair[0]] for pair in pairs]
#     return ordereds

# # kwlee
# def track_str(track):
#     return f"{track.track_id}:{track.age}({track.hits})"
# def row_str(row):
#     return [round(v,3) if v < 10 else '' for v in row]

# # kwlee
# def get_track_info(idx, track_indices, tracks):
#     return tracks[idx], track_indices.index(idx)

def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections,
                    track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    # 'time_since_update'별로 task를 grouping 한다.
    leveled_track_indices = defaultdict(list)
    for idx in track_indices:
        level = tracks[idx].time_since_update - 1
        if level < cascade_depth:
            leveled_track_indices[level].append(idx)

    # 'time_since_update'별로 'min_cost_matching'을 실시하여 그 결과를 merge시킨다.
    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices,
                    gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    # gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # kwlee
    gating_threshold = kalman_filter.chi2inv95[gating_dim] * 3
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)

        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix



    # kwlee
_DIST_THRESHOLD = 20
def matching_distance(dist_matrix, tracks, detections, track_indices):
    matches = []
    unmatched_detections = list(range(len(detections)))
    ndets = len(detections)
    if ndets <= 0:
        return matches, track_indices, unmatched_detections

    unmatched_tracks = [i for i in track_indices if tracks[i].time_since_update > 1]
    track_indices = [i for i in track_indices if tracks[i].time_since_update <= 1]
    for tidx in track_indices:
        track = tracks[tidx]
        dists = dist_matrix[tidx,:]
        if ndets > 2:
            idxes = np.argpartition(dists, 2)[:2]
            v1, v2 = tuple(dists[idxes])
        elif ndets == 2:
            idxes = [0, 1] if dists[0] <= dists[1] else [1, 0]
            v1, v2 = tuple(dists[idxes])
        else:
            idxes = [0, -1]
            v1, v2 = dists[0], 9999

        if v1 < _DIST_THRESHOLD and v1*2 < v2:
            det_idx = idxes[0]
            dists2 = dist_matrix[track_indices,det_idx]
            # dists2 = dist_matrix[:,det_idx]
            cnt = len(dists2[np.where(dists2 < _DIST_THRESHOLD)])
            if cnt <= 1:
                matches.append((tidx, idxes[0]))
                unmatched_detections.remove(idxes[0])
                continue
        unmatched_tracks.append(tidx)
    
    return matches, unmatched_tracks, unmatched_detections