# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from collections import defaultdict
import enum

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
    gating_dim = 2 if only_position else 4
    # gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # kwlee
    gating_threshold = kalman_filter.chi2inv95[gating_dim] * 4
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)

        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

# kwlee
def find_bottom2_indexes(values, indices):
    count = len(indices)
    if count > 2:
        rvals = [values[i] for i in indices]
        idx1, idx2 = np.argpartition(rvals, 2)[:2]
        return [indices[idx1], indices[idx2]]
    elif count == 2:
        return [indices[0], indices[1]] if values[indices[0]] <= values[indices[1]] else [indices[1], indices[0]]
    else:
        return [indices[0], None]

# kwlee
_CLOSE_DIST_THRESHOLD = 21
_INFINIT_DIST = 9999
def matching_by_close_distance(dist_matrix, tracks, detections, track_indices, detection_indices=None):
    if not detection_indices:
        detection_indices = list(range(len(detections)))
        
    if len(detection_indices) <= 0:
        return [], track_indices, detection_indices

    matches = []
    unmatched_tracks = track_indices.copy()
    unmatched_detections = detection_indices.copy()
    dist_matrix = dist_matrix.copy()
    for tidx in track_indices:
        track = tracks[tidx]

        dists = dist_matrix[tidx,:]
        idxes = find_bottom2_indexes(dists, detection_indices)
        if idxes[1]:
            v1, v2 = tuple(dists[idxes])
        else:
            v1, v2 = dists[idxes[0]], _INFINIT_DIST

        if v1 < _CLOSE_DIST_THRESHOLD and v1*2 < v2:
            det_idx = idxes[0]
            dists2 = dist_matrix[unmatched_tracks, det_idx]
            cnt = len(dists2[np.where(dists2 < _CLOSE_DIST_THRESHOLD)])
            if cnt <= 1:
                matches.append((tidx, idxes[0]))
                unmatched_tracks.remove(tidx)
                unmatched_detections.remove(idxes[0])
                dist_matrix[:,idxes[0]] = _INFINIT_DIST
                continue
    
    return matches, unmatched_tracks, unmatched_detections

_COMBINED_METRIC_THRESHOLD = 0.55
_COMBINED_METRIC_THRESHOLD_4L = 0.45
_COMBINED_DIST_THRESHOLD = 75
_COMBINED_DIST_THRESHOLD_4_LARGE = 310
_COMBINED_INFINITE = 9.99
import math
def combine_cost_matrices(metric_costs, dist_costs, tracks, detections):
    # dists_mod = dist_costs / _COMBINED_DIST_THRESHOLD

    # time_since_update 에 따른 가중치 보정
    weights = list(map(lambda t: math.log10(t.time_since_update), tracks))
    weighted_dist_costs = dist_costs.copy()
    for tidx, track in enumerate(tracks):
        if weights[tidx] > 0:
            weighted_dist_costs[tidx,:] = dist_costs[tidx,:] * weights[tidx]
    dists_mod = weighted_dist_costs / _COMBINED_DIST_THRESHOLD

    # # temporary lost 횟수를 통한 가중치 계산
    # tsu = np.array([0.2*t.time_since_update/20 for t in tracks])

    matrix = np.zeros((len(tracks), len(detections)))
    invalid = np.zeros((len(tracks), len(detections)), dtype=bool)
    for didx, det in enumerate(detections):
        det = detections[didx]

        if det.tlwh[2] >= 150 and det.tlwh[3] >= 150: # large detections
            # detection 크기가 크면 metric cost에 많은 가중치를 주어, 외형을 보다 많이 고려한다.
            # 또한 외형에 많은 가중치를 주기 때문에 gate용 distance 한계도 넉넉하게 (300) 주는 대신,
            # gate용 metric thresholds는 다른 경우보다 작게 (0.45) 준다.
            matrix[:,didx] = 0.8*metric_costs[:,didx] + 0.2*dists_mod[:,didx]
            invalid[:,didx] = np.logical_or(metric_costs[:,didx] > _COMBINED_METRIC_THRESHOLD_4L,
                                            weighted_dist_costs[:,didx] > _COMBINED_DIST_THRESHOLD_4_LARGE)
        elif det.tlwh[2] >= 25 and det.tlwh[3] >= 25: # medium detections
            matrix[:,didx] = 0.7*metric_costs[:,didx] + 0.3*dists_mod[:,didx]
            invalid[:,didx] = np.logical_or(metric_costs[:,didx] > _COMBINED_METRIC_THRESHOLD,
                                            weighted_dist_costs[:,didx] > _COMBINED_DIST_THRESHOLD)
        else:
            # detection의 크기가 작으면 외형을 이용한 검색이 의미가 작으므로, track과 detection사이의 거리
            # 정보에 보다 많은 가중치를 부여한다.
            matrix[:,didx] = 0.2*metric_costs[:,didx] + 0.8*dists_mod[:,didx]
            invalid[:,didx] = np.logical_or(metric_costs[:,didx] > _COMBINED_METRIC_THRESHOLD,
                                            weighted_dist_costs[:,didx] > _COMBINED_DIST_THRESHOLD)
    matrix[invalid] = _COMBINED_INFINITE

    return matrix

def _remove_by_index(list, idx):
    removed = list[idx]
    return removed, (list[:idx] + list[idx+1:])

def matching_by_total_cost(cost_matrix, track_indices, detection_indices, threshold=0.7):
    if len(track_indices) <= 1 and len(detection_indices) <= 1:
        if cost_matrix[track_indices[0], detection_indices[0]] <= threshold:
            return [(track_indices[0], detection_indices[0])], [], []
        else:
            return [], track_indices, detection_indices
    elif len(detection_indices) <= 1:   # track만 여러개
        reduced = cost_matrix[:,detection_indices[0]][track_indices]
        tidx = np.argmin(reduced)
        if reduced[tidx] <= threshold:
            matched_track, unmatched_tracks = _remove_by_index(track_indices, tidx)
            return [(matched_track, detection_indices[0])], unmatched_tracks, []
        else:
            return [], track_indices, detection_indices
    elif len(track_indices) <= 1:       # detection만 여러개
        reduced = cost_matrix[track_indices[0],:][detection_indices]
        didx = np.argmin(reduced)
        if reduced[didx] <= threshold:
            matched_det, unmatched_dets = _remove_by_index(detection_indices, didx)
            return [(track_indices[0], matched_det)], [], unmatched_dets
        else:
            return [], track_indices, detection_indices

    matrix = cost_matrix[np.ix_(track_indices, detection_indices)]
    indices = linear_sum_assignment(matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

    matches = []
    unmatched_tracks = track_indices.copy()
    unmatched_detections = detection_indices.copy()
    for i, j in indices:
        tidx = track_indices[i]
        didx = detection_indices[j]
        if cost_matrix[tidx, didx] <= threshold:
            matches.append((tidx, didx))
            unmatched_tracks.remove(tidx)
            unmatched_detections.remove(didx)
    
    return matches, unmatched_tracks, unmatched_detections