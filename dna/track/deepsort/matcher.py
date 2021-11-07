
import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

import dna
from dna import Box, get_logger
from .utils import find_overlaps, find_overlaps_threshold, project


_logger = get_logger("dna.track.deep_sort")

_HUGE = 300
_LARGE = 150
_MEDIUM = 30
_COMBINED_METRIC_THRESHOLD_4S = 0.89
_COMBINED_METRIC_THRESHOLD_4M = 0.55
_COMBINED_METRIC_THRESHOLD_4L = 0.40
_COMBINED_DIST_THRESHOLD_4S = 75
_COMBINED_DIST_THRESHOLD_4M = 150
_COMBINED_DIST_THRESHOLD_4_LARGE = 310
_COMBINED_INFINITE = 9.99

import math
def combine_cost_matrices(metric_costs, dist_costs, tracks, detections):
    # time_since_update 에 따른 가중치 보정
    weights = list(map(lambda t: math.log10(t.time_since_update)+1, tracks))
    weighted_dist_costs = dist_costs.copy()
    for tidx, track in enumerate(tracks):
        if weights[tidx] > 0:
            weighted_dist_costs[tidx,:] = dist_costs[tidx,:] * weights[tidx]
    dists_mod = weighted_dist_costs / _COMBINED_DIST_THRESHOLD_4S

    matrix = np.zeros((len(tracks), len(detections)))
    invalid = np.zeros((len(tracks), len(detections)), dtype=bool)
    for didx, det in enumerate(detections):
        det = detections[didx]

        if det.tlwh[2] >= _LARGE and det.tlwh[3] >= _LARGE: # large detections
            # detection 크기가 크면 metric cost에 많은 가중치를 주어, 외형을 보다 많이 고려한다.
            # 또한 외형에 많은 가중치를 주기 때문에 gate용 distance 한계도 넉넉하게 (300) 주는 대신,
            # gate용 metric thresholds는 다른 경우보다 작게 (0.45) 준다.
            matrix[:,didx] = 0.8*metric_costs[:,didx] + 0.2*dists_mod[:,didx]
            invalid[:,didx] = np.logical_or(metric_costs[:,didx] > _COMBINED_METRIC_THRESHOLD_4L,
                                            weighted_dist_costs[:,didx] > _COMBINED_DIST_THRESHOLD_4_LARGE)
        elif det.tlwh[2] >= _MEDIUM and det.tlwh[3] >= _MEDIUM: # medium detections
            matrix[:,didx] = 0.7*metric_costs[:,didx] + 0.3*dists_mod[:,didx]
            invalid[:,didx] = np.logical_or(metric_costs[:,didx] > _COMBINED_METRIC_THRESHOLD_4M,
                                            weighted_dist_costs[:,didx] > _COMBINED_DIST_THRESHOLD_4M)
        else:
            # detection의 크기가 작으면 외형을 이용한 검색이 의미가 작으므로, track과 detection사이의 거리
            # 정보에 보다 많은 가중치를 부여한다.
            matrix[:,didx] = 0.2*metric_costs[:,didx] + 0.8*dists_mod[:,didx]
            invalid[:,didx] = np.logical_or(metric_costs[:,didx] > _COMBINED_METRIC_THRESHOLD_4S,
                                            weighted_dist_costs[:,didx] > _COMBINED_DIST_THRESHOLD_4S)
    matrix[invalid] = _COMBINED_INFINITE

    return matrix


def matching_by_excl_best(dist_matrix, threshold, track_indices, detection_indices):
    def find_best2_indices(values, indices):
        count = len(indices)
        if count > 2:
            rvals = [values[i] for i in indices]
            idx1, idx2 = np.argpartition(rvals, 2)[:2]
            return [indices[idx1], indices[idx2]]
        elif count == 2:
            return [indices[0], indices[1]] if values[indices[0]] <= values[indices[1]] else [indices[1], indices[0]]
        else:
            return [indices[0], None]

    matches = []
    unmatched_tracks = track_indices.copy()
    unmatched_detections = detection_indices.copy()
    for tidx in track_indices:
        dists = dist_matrix[tidx,:]
        idxes = find_best2_indices(dists, unmatched_detections)
        if idxes[1]:
            v1, v2 = tuple(dists[idxes])
        else:
            v1, v2 = dists[idxes[0]], threshold+0.01

        if v1 < threshold and (v2 >= threshold or v1*2 < v2):
            didx = idxes[0]
            dists2 = dist_matrix[unmatched_tracks, didx]
            cnt = len(dists2[np.where(dists2 < threshold)])
            if cnt <= 1:
                matches.append((tidx, didx))
                unmatched_tracks.remove(tidx)
                unmatched_detections.remove(didx)
                if len(unmatched_detections) == 0:
                    break
                else:
                    continue
    
    return matches, unmatched_tracks, unmatched_detections


def matching_by_hungarian(cost_matrix, threshold, track_indices, detection_indices):
    def _remove_by_index(list, idx):
        removed = list[idx]
        return removed, (list[:idx] + list[idx+1:])

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


def matching_by_overlap(tracks, detections, ov_match, track_indices, detection_indices):
    matches = []
    unmatched_tracks = track_indices.copy()
    dets = [Box.from_tlbr(detections[didx].to_tlbr()) for didx in detection_indices]
    indices = list(range(len(dets)))

    for tidx in track_indices:
        box = Box.from_tlbr(tracks[tidx].to_tlbr())
        overlaps = find_overlaps(box, dets, ov_match, indices)
        if len(overlaps) > 0:
            best_ov = sorted(overlaps, key=lambda ov: max(ov[1]), reverse=True)[0]
            didx = detection_indices[best_ov[0]]

            matches.append((tidx, didx))
            unmatched_tracks.remove(tidx)
            indices.remove(best_ov[0])
            if len(indices) == 0:
                break

    return matches, unmatched_tracks, list([detection_indices[idx] for idx in indices])

def remove_overlaps(targets, detections, threshold, target_indices, candidate_indices):
    candidate_boxes = [Box.from_tlbr(detections[didx].to_tlbr()) for didx in candidate_indices]

    overlaps = set()
    for tidx in target_indices:
        box = Box.from_tlbr(targets[tidx].to_tlbr())
        ovs = find_overlaps_threshold(box, candidate_boxes, threshold)
        if len(ovs) > 0:
            ov_idxes = [candidate_indices[idx] for idx in project(ovs, 0)]
            overlaps.update(ov_idxes)
            if _logger.isEnabledFor(logging.DEBUG):
                ov_str = ",".join([f"{candidate_indices[i]}({max(r):.2f})" for i, r in ovs])
                _logger.debug((f"remove hot-track's overlaps: "
                                f"track's det={tidx}, det={ov_str}, frame={dna.DEBUG_FRAME_IDX}"))
    non_overlaps = set(candidate_indices) - overlaps
    return list(non_overlaps)

def delete_overlapped_tentative_tracks(tracks, threshold):
    confirmed_tracks = [i for i, t in enumerate(tracks) if t.is_confirmed() and t.time_since_update == 1 and not t.is_deleted()]
    unconfirmed_tracks = [i for i, t in enumerate(tracks) if not t.is_confirmed() and not t.is_deleted()]

    if len(confirmed_tracks) > 0 and len(unconfirmed_tracks) > 0:
        track_boxes = [Box.from_tlbr(track.to_tlbr()) for track in tracks]

        suriveds = []
        for uc_idx in unconfirmed_tracks:
            ovs = find_overlaps_threshold(track_boxes[uc_idx], track_boxes, threshold, confirmed_tracks)
            if len(ovs) > 0:
                tracks[uc_idx].mark_deleted()
                if _logger.isEnabledFor(logging.DEBUG):
                    uc_track_id = tracks[uc_idx].track_id
                    ov_track_id = tracks[ovs[0][0]].track_id
                    ov_ratio = max(ovs[0][1])
                    _logger.debug((f"delete tentative track[{uc_track_id}] because it is too close to track[{ov_track_id}], "
                                    f"ratio={ov_ratio:.2f}, frame={dna.DEBUG_FRAME_IDX}"))
            else:
                suriveds.append(uc_idx)