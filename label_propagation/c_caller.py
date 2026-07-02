import os
import sys
import numpy as np
import ctypes

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_with_indent, INDENT_LOG


class CCaller:

    def __init__(self, c_lib_filename):
        self.caller = ctypes.CDLL(c_lib_filename) if os.path.isfile(c_lib_filename) else None

    def is_available(self, verbose=True):
        # if verbose and self.caller is None:
        #     print_with_indent("Warning: C caller is not loaded, algorithms may be 5-10x slower.", indent=INDENT_LOG)
        return self.caller is not None

    @staticmethod
    def get_argtype(dtype, ndim):
        return np.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, flags="C_CONTIGUOUS")


class SpeakerCCaller(CCaller):

    def get_active_map_c(self, subtitle_timestamp, face_timestamp, subtitle_code, face_code, face_score, mask, roles,
                         face_radius, face_threshold):
        subtitles, faces = subtitle_timestamp.shape[0], face_timestamp.shape[0]
        c_func = self.caller.get_active_map
        c_func.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, self.get_argtype(np.int64, 2),
                           self.get_argtype(np.int64, 2), self.get_argtype(np.int32, 1), self.get_argtype(np.int32, 1),
                           self.get_argtype(np.float32, 1), self.get_argtype(np.uint8, 1), ctypes.c_int, ctypes.c_float]
        c_func.restype = self.get_argtype(np.uint8, 1)
        c_result = c_func(subtitles, faces, roles, subtitle_timestamp, face_timestamp, subtitle_code, face_code,
                          face_score, mask, face_radius, face_threshold)
        active_map = np.ctypeslib.as_array(ctypes.cast(c_result, ctypes.POINTER(ctypes.c_ubyte)),
                                           shape=(subtitles, roles,))
        return active_map

    def get_connected_components_c(self, similarity, active_idx, sim_threshold, search_radius):
        subtitles, active_subtitles = similarity.shape[0], len(active_idx)
        c_func = self.caller.get_connected_components
        c_func.argtypes = [ctypes.c_int, ctypes.c_int, self.get_argtype(np.float32, 2),
                           self.get_argtype(np.int32, 1), ctypes.c_float, ctypes.c_int]
        c_func.restype = self.get_argtype(np.int32, 1)
        c_result = c_func(subtitles, active_subtitles, similarity, active_idx, sim_threshold, search_radius)
        result = np.ctypeslib.as_array(ctypes.cast(c_result, ctypes.POINTER(ctypes.c_int)),
                                       shape=(2, active_subtitles,))
        component_id, component_size = result[0, :], result[1, :]
        component_size = component_size[component_size > 0]
        return component_id, component_size

    def get_propagated_labels_c(self, similarity, active_map, active_idx, current_label, subtitle_length,
                                sim_threshold, max_candidates):
        subtitles, roles, active_subtitles = similarity.shape[0], active_map.shape[1], len(active_idx)
        c_func = self.caller.get_propagated_labels
        c_func.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, self.get_argtype(np.float32, 2),
                           self.get_argtype(np.uint8, 2), self.get_argtype(np.int32, 1), self.get_argtype(np.int32, 1),
                           self.get_argtype(np.int32, 1), ctypes.c_float, ctypes.c_int]
        c_func.restype = self.get_argtype(np.float32, 1)
        c_result = c_func(subtitles, roles, active_subtitles, similarity, active_map, active_idx, current_label,
                          subtitle_length, sim_threshold, max_candidates)
        result = np.ctypeslib.as_array(ctypes.cast(c_result, ctypes.POINTER(ctypes.c_float)),
                                       shape=(active_subtitles, max_candidates * 2 + 1,))
        return result.tolist()

    def compute_rel_info_c(self, similarity, active_map, current_label, idx_list_length, idx_list_all, base_threshold,
                           delta_threshold, display_info):
        subtitles, roles = similarity.shape[0], active_map.shape[1]
        components, total_length = len(idx_list_length), len(idx_list_all)
        c_func = self.caller.compute_rel_info
        c_func.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, self.get_argtype(np.float32, 2),
                           self.get_argtype(np.uint8, 2), self.get_argtype(np.int32, 1), self.get_argtype(np.int32, 1),
                           self.get_argtype(np.int32, 1), ctypes.c_float, ctypes.c_float, ctypes.c_ubyte]
        c_func.restype = self.get_argtype(np.float32, 1)
        c_result = c_func(subtitles, roles, components, total_length, similarity, active_map, current_label,
                          np.array(idx_list_length, dtype=np.int32), np.array(idx_list_all, dtype=np.int32),
                          base_threshold, delta_threshold, display_info)
        unit_size = components * roles
        total_size = unit_size * 6 + total_length * 2
        result = np.ctypeslib.as_array(ctypes.cast(c_result, ctypes.POINTER(ctypes.c_float)),
                                       shape=(total_size,))
        active_idx_count = result[0: unit_size].astype(np.int32).reshape((components, roles)).tolist()
        max_max = result[unit_size: unit_size * 2].reshape((components, roles)).tolist()
        max_median = result[unit_size * 2: unit_size * 3].reshape((components, roles)).tolist()
        median_max = result[unit_size * 3: unit_size * 4].reshape((components, roles)).tolist()
        mean_avg_top = result[unit_size * 4: unit_size * 5].reshape((components, roles)).tolist()
        is_high_similarity = result[unit_size * 5: unit_size * 6].astype(np.uint8).reshape((components, roles)).tolist()
        global_sim = result[unit_size * 6: total_size].reshape((total_length, 2)) if display_info else None
        return active_idx_count, max_max, max_median, median_max, mean_avg_top, is_high_similarity, global_sim
