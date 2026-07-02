import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_module_title, print_with_indent, INDENT_MODULE, INDENT_INFO, INDENT_LOG
from c_caller import SpeakerCCaller
from label_structure import LabelSet, LabelItem, Component


class GraphAlgorithm:

    C_LIB_FILENAME = "./label_propagation/c_functions.so"
    NO_C_LIB_WARNING = lambda function_name: "Warining: C function for {:s}() was not loaded; program may be 5-10x slow!".format(function_name)

    def __init__(self, c_lib_filename=None):
        c_lib_filename = c_lib_filename if c_lib_filename is not None else GraphAlgorithm.C_LIB_FILENAME
        self.c_caller = SpeakerCCaller(c_lib_filename)

    def compute_similarity(self, embedding):
        print_with_indent("Start computing voiceprint similarity...", indent=INDENT_INFO)
        start_time = time.time()
        normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        similarity = np.dot(normalized_embedding, normalized_embedding.T)
        similarity = np.nan_to_num(similarity, copy=False)
        np.fill_diagonal(similarity, 0)
        print_with_indent("Finished computing voiceprint similarity; {:0.6f} seconds elapsed.".format(
            time.time() - start_time), indent=INDENT_INFO)
        return similarity

    @staticmethod
    def _get_active_map_py(subtitle_timestamp, face_timestamp, subtitle_code, face_code, face_score, mask, characters,
                           face_radius, face_threshold):
        subtitles, faces = len(subtitle_timestamp), len(face_timestamp)
        active_map = np.zeros((subtitles, characters), dtype=np.int32)
        character_count = [0] * characters
        left_face_idx, right_face_idx = 0, 0
        left_subtitle_idx, right_subtitle_idx = 0, 0
        for subtitle_idx in range(subtitles):
            while right_face_idx < faces and face_timestamp[right_face_idx][0] <= subtitle_timestamp[
                subtitle_idx][1] + face_radius:
                if 0 <= face_code[right_face_idx] < characters and face_score[right_face_idx] >= face_threshold:
                    character_count[face_code[right_face_idx]] += 1
                right_face_idx += 1
            while left_face_idx < faces and face_timestamp[left_face_idx][1] < subtitle_timestamp[
                subtitle_idx][0] - face_radius:
                if 0 <= face_code[left_face_idx] < characters and face_score[left_face_idx] >= face_threshold:
                    character_count[face_code[left_face_idx]] -= 1
                left_face_idx += 1
            while right_subtitle_idx < subtitles and subtitle_timestamp[right_subtitle_idx][0] <= \
                    subtitle_timestamp[subtitle_idx][1] + face_radius:
                if 0 <= subtitle_code[right_subtitle_idx] < characters:
                    character_count[subtitle_code[right_subtitle_idx]] += 1
                right_subtitle_idx += 1
            while left_subtitle_idx < subtitles and subtitle_timestamp[left_subtitle_idx][1] < subtitle_timestamp[
                subtitle_idx][0] - face_radius:
                if 0 <= subtitle_code[left_subtitle_idx] < characters:
                    character_count[subtitle_code[left_subtitle_idx]] -= 1
                left_subtitle_idx += 1
            if mask[subtitle_idx]:
                active_map[subtitle_idx, :] = np.array(character_count)
        return (active_map > 0).astype(np.uint8)

    def _get_active_map(self, info_set, curr_label_set, face_radius, face_score_threshold, mask=None):
        subtitle_timestamp, face_timestamp = info_set["subtitle_timestamp"], info_set["face_timestamp"]
        face_code, face_score = info_set["face_code"], info_set["face_score"]
        subtitle_code = curr_label_set.code_arr
        subtitles, faces = len(subtitle_timestamp), len(face_timestamp)
        assert subtitles == len(subtitle_code), "Error: the numbers of subtitles and utterance labels do not match!"
        assert faces == len(face_code), "Error: the numbers of faces and character labels do not match!"
        if mask is None:
            mask = [1] * subtitles
        characters = curr_label_set.get_known_characters()
        func = self.c_caller.get_active_map_c if self.c_caller.is_available() else GraphAlgorithm._get_active_map_py
        if func == GraphAlgorithm._get_active_map_py:
            print_with_indent(GraphAlgorithm.NO_C_LIB_WARNING("get_active_map"), indent=INDENT_LOG)
        active_map = func(np.array(subtitle_timestamp, dtype=np.int64), np.array(face_timestamp, dtype=np.int64),
                          np.array(subtitle_code, dtype=np.int32), np.array(face_code, dtype=np.int32),
                          np.array(face_score, dtype=np.float32), np.array(mask, dtype=np.uint8),
                          characters, face_radius, face_score_threshold)
        active_map[:, curr_label_set.get_global_code_list()] = 1
        return active_map

    @staticmethod
    def _get_connected_components_py(similarity, active_idx, sim_threshold, search_radius):
        n, m = similarity.shape[0], len(active_idx)
        start_pos, end_pos = np.zeros_like(active_idx), np.zeros_like(active_idx)
        left_cursor, right_cursor = 0, 0
        for i in range(m):
            while active_idx[left_cursor] < active_idx[i] - search_radius:
                left_cursor += 1
            while right_cursor < m and active_idx[right_cursor] <= active_idx[i] + search_radius:
                right_cursor += 1
            start_pos[i], end_pos[i] = left_cursor, right_cursor
        components = 0
        component_id = np.ones(m, dtype=np.int32) * (-1)
        component_size = np.zeros(m, dtype=np.int32)
        queue = np.zeros(m, dtype=np.int32)
        for i in range(len(active_idx)):
            if component_id[i] == -1:
                head, tail = 0, 1
                component_id[i] = components
                queue[0] = i
                while head < tail:
                    start_idx, end_idx = i, m
                    if search_radius > 0:
                        start_idx, end_idx = max(i, start_pos[queue[head]]), min(m, end_pos[queue[head]])
                    idx = np.where(np.logical_and(component_id[start_idx: end_idx] == -1,
                                                  similarity[active_idx[queue[head]]][
                                                      active_idx[start_idx: end_idx]] >= sim_threshold))[0]
                    component_id[start_idx + idx] = components
                    queue[tail: tail + len(idx)] = start_idx + idx
                    tail += len(idx)
                    head += 1
                component_size[components] = tail
                components += 1
        component_size = component_size[:components]
        return component_id, component_size

    def _get_max_connected_components(self, similarity, active_idx, sim_threshold, search_radius=0):
        sub_similarity = similarity[active_idx, :][:, active_idx]
        avg_similarity = np.mean(sub_similarity, axis=1)
        func = self.c_caller.get_connected_components_c if self.c_caller.is_available() else GraphAlgorithm._get_connected_components_py
        if func == GraphAlgorithm._get_connected_components_py:
            print_with_indent(GraphAlgorithm.NO_C_LIB_WARNING("get_connected_components"), indent=INDENT_LOG)
        component_id, component_size = func(similarity, np.array(active_idx, dtype=np.int32), sim_threshold, search_radius)
        max_idx_list = np.argwhere(component_size == np.amax(component_size)).flatten().tolist()
        best_idx_list, max_avg_similarity = None, -1
        for _ in range(len(max_idx_list)):
            item_idx_list = [item_idx for item_idx in range(len(component_id)) if component_id[item_idx] == max_idx_list[_]]
            idx_list = [active_idx[idx] for idx in item_idx_list]
            avg_similarity_ = np.mean(avg_similarity[item_idx_list])
            if max_avg_similarity < avg_similarity_:
                best_idx_list, max_avg_similarity = idx_list, avg_similarity_
        return best_idx_list

    def get_most_confident_component(self, similarity, active_idx, target_utterances):
        min_threshold, max_threshold = 0.4, 1.0
        confident_cc = None
        while max_threshold - min_threshold >= 1e-8:
            curr_threshold = (min_threshold + max_threshold) / 2
            max_cc = self._get_max_connected_components(similarity, active_idx, curr_threshold, 0)
            curr_utterances = len(max_cc)
            if curr_utterances <= target_utterances:
                confident_cc = max_cc
                if curr_utterances == target_utterances:
                    break
                max_threshold = curr_threshold
            else:
                min_threshold = curr_threshold
        assert confident_cc is not None, "Error: no connected component is found!"
        return confident_cc

    def _get_components_by_similarity(self, similarity, active_idx, character_name, sim_threshold, size_threshold, subtitle_radius):
        func = self.c_caller.get_connected_components_c if self.c_caller.is_available() else GraphAlgorithm._get_connected_components_py
        if func == GraphAlgorithm._get_connected_components_py:
            print_with_indent(GraphAlgorithm.NO_C_LIB_WARNING("get_connected_components"), indent=INDENT_LOG)
        component_id, component_size = func(similarity, active_idx, sim_threshold, subtitle_radius)
        new_component_list, new_component_size = [], []
        for component_idx in range(len(component_size)):
            if component_size[component_idx] < size_threshold:
                continue
            sub_idx = np.where(component_id == component_idx)[0]
            new_component_list.append(Component(active_idx[sub_idx].tolist(), character_name=character_name))
            new_component_size.append(len(sub_idx))
        sorted_idx = [idx[0] for idx in sorted(enumerate(new_component_size), key=lambda _: -_[1])]
        return [new_component_list[idx] for idx in sorted_idx]

    @staticmethod
    def _get_propagated_labels_py(similarity, active_map, active_idx, current_label, subtitle_length, sim_threshold, max_candidates):
        subtitles, characters = similarity.shape[0], active_map.shape[1]
        character_active_idx, character_labeled_idx = [[] for _ in range(characters)], [[] for _ in range(characters)]
        for character_idx in range(characters):
            character_active_idx[character_idx] = np.where(active_map[:, character_idx] == 1)[0].tolist()
            character_labeled_idx[character_idx] = np.where(current_label == character_idx)[0].tolist()
        character_unchanged = [False] * characters
        new_label_list = []
        while True:
            sim_avg = np.zeros((subtitles, characters), dtype=np.float32)
            for character_idx in range(characters):
                if character_unchanged[character_idx]:
                    continue
                new_idx = character_active_idx[character_idx]
                old_idx = CcOperations._filter_idx_list_by_subtitle_length(character_labeled_idx[character_idx], subtitle_length)
                if len(old_idx) > 0:
                    sim_avg[new_idx, character_idx] = CcOperations._get_average(similarity[new_idx, :][:, old_idx])
                character_unchanged[character_idx] = True
            new_labeled_idx = []
            for subtitle_idx in active_idx:
                added_similarity = min(max(1 - subtitle_length[subtitle_idx] / 1000, 0), 0.5) * 0.2
                sim_avg[subtitle_idx, :] += added_similarity
                sorted_character_idx = np.argsort(-sim_avg[subtitle_idx, :])
                if sim_avg[subtitle_idx, sorted_character_idx[0]] < sim_threshold:
                    continue
                new_label_item, new_label_code, candidates = [], -1, 0
                for character_idx in sorted_character_idx:
                    if active_map[subtitle_idx, character_idx] == 1 and sim_avg[subtitle_idx, character_idx] > 0:
                        if new_label_code == -1:
                            new_label_item, new_label_code = [subtitle_idx], character_idx
                        new_label_item.extend([character_idx, sim_avg[subtitle_idx, character_idx]])
                        candidates += 1
                        if candidates == max_candidates:
                            break
                if new_label_code >= 0:
                    new_label_list.append(new_label_item)
                    character_labeled_idx[new_label_code].append(subtitle_idx)
                    new_labeled_idx.append(subtitle_idx)
                    character_unchanged[new_label_code] = False
            if len(new_labeled_idx) < 1:
                break
            active_idx = np.setdiff1d(active_idx, np.array(new_labeled_idx))
        return new_label_list

    def _get_propagated_label_set(self, info_set, curr_label_set, similarity, sim_threshold_base, face_radius, face_score_threshold):
        active_map = self._get_active_map(info_set, curr_label_set, face_radius, face_score_threshold)
        active_idx = np.array(curr_label_set.get_unlabeled_utterance_idx(), dtype=np.int32)
        current_code_arr = np.copy(curr_label_set.code_arr)
        subtitle_timestamp = np.array(info_set["subtitle_timestamp"], dtype=np.int64)
        subtitle_length = (subtitle_timestamp[:, 1] - subtitle_timestamp[:, 0]).astype(np.int32)
        max_candidates = 5
        func = self.c_caller.get_propagated_labels_c if self.c_caller.is_available() else GraphAlgorithm._get_propagated_labels_py
        if func == GraphAlgorithm._get_propagated_labels_py:
            print_with_indent(GraphAlgorithm.NO_C_LIB_WARNING("get_propagated_labels"), indent=INDENT_LOG)
        new_label_list = func(similarity, active_map, active_idx, current_code_arr, subtitle_length, sim_threshold_base,
                              max_candidates)
        new_label_item_list = []
        for idx in range(len(new_label_list)):
            subtitle_idx, rel_info_ = int(new_label_list[idx][0]), new_label_list[idx][1:]
            if subtitle_idx == -1:
                break
            rel_info = {"_list": [], "_dict": {}}
            for l in range(min(len(rel_info_) // 2, max_candidates)):
                code, sim_avg = int(rel_info_[l * 2]), float(rel_info_[l * 2 + 1])
                if l == 0 or sim_avg >= sim_threshold_base:
                    name = curr_label_set.get_name_by_code(code)
                    rel_info["_list"].append(name)
                    rel_info["_dict"].update({name: sim_avg})
                else:
                    break
            new_label_item_list.append(LabelItem(subtitle_idx, rel_info=rel_info))
        return new_label_item_list

    @staticmethod
    def _compute_rel_info_py(similarity, active_map, current_label, idx_list_length, idx_list_all, base_threshold,
                             delta_threshold, display_info):
        subtitles, characters = similarity.shape[0], active_map.shape[1]
        components, total_length = len(idx_list_length), len(idx_list_all)
        known_code = np.unique(current_label[current_label >= 0])
        active_idx_count = np.zeros((components, characters), dtype=np.int32)
        max_max = np.zeros((components, characters), dtype=np.float32)
        max_median = np.zeros((components, characters), dtype=np.float32)
        median_max = np.zeros((components, characters), dtype=np.float32)
        mean_avg_top = np.zeros((components, characters), dtype=np.float32)
        is_high_similarity = np.zeros((components, characters), dtype=np.uint8)
        global_sim = np.zeros((total_length, 2), dtype=np.float32) if display_info else None
        for code in known_code:
            labeled_idx_list = np.where(current_label == code)[0]
            if len(labeled_idx_list) == 0:
                continue
            cross_similarity = similarity[idx_list_all, :][:, labeled_idx_list]
            col_max, col_median = np.max(cross_similarity, axis=1), np.median(cross_similarity, axis=1)
            avg_top = CcOperations._get_average(cross_similarity)
            idx_offset = 0
            for cp_idx in range(components):
                idx_list_idx = np.s_[idx_offset: idx_offset + idx_list_length[cp_idx]]
                subtitle_idx_list = idx_list_all[idx_list_idx]
                active_idx_count[cp_idx][code] = np.count_nonzero(active_map[subtitle_idx_list, code] > 0)
                max_max[cp_idx][code], max_median[cp_idx][code], median_max[cp_idx][code] = \
                    np.max(col_max[idx_list_idx]), np.max(col_median[idx_list_idx]), np.median(col_max[idx_list_idx])
                mean_avg_top[cp_idx][code] = np.mean(avg_top[idx_list_idx])
                is_high_similarity[cp_idx][code] = CcOperations._is_high_similarity(
                    active_idx_count[cp_idx][code] / idx_list_length[cp_idx], mean_avg_top[cp_idx][code],
                    base_threshold, delta_threshold)
                if display_info:
                    if is_high_similarity[cp_idx][code]:
                        global_sim[idx_list_idx, 0] = np.maximum(global_sim[idx_list_idx, 0], col_max[idx_list_idx])
                    global_sim[idx_list_idx, 1] = np.maximum(global_sim[idx_list_idx, 1], col_max[idx_list_idx])
                idx_offset += idx_list_length[cp_idx]
        return active_idx_count.tolist(), max_max.tolist(), max_median.tolist(), median_max.tolist(), mean_avg_top.tolist(), is_high_similarity.tolist(), global_sim

    def _get_similarity_information(self, info_set, curr_label_set, similarity, component_list,
                                    base_threshold, delta_threshold, face_radius, face_score_threshold, display_info=False, compute_info=False):
        components = len(component_list)
        labeled_idx_mask_all = curr_label_set.get_mask_by_component_list(component_list)
        active_map = self._get_active_map(info_set, curr_label_set, face_radius, face_score_threshold, mask=labeled_idx_mask_all)
        code_arr = curr_label_set.code_arr
        known_code = np.unique(code_arr[code_arr >= 0])
        idx_list_length, idx_list_all = [], []
        for component in component_list:
            component_idx_list = component.get_idx_list()
            idx_list_length.append(len(component_idx_list))
            idx_list_all.extend(component_idx_list)
        func = self.c_caller.compute_rel_info_c if self.c_caller.is_available() else GraphAlgorithm._compute_rel_info_py
        if func == GraphAlgorithm._compute_rel_info_py:
            print_with_indent(GraphAlgorithm.NO_C_LIB_WARNING("compute_rel_info"), indent=INDENT_LOG)
        active_idx_count, max_max, max_median, median_max, mean_avg_top, is_high_similarity, global_sim_arr = \
            func(similarity, active_map, code_arr, idx_list_length, idx_list_all, base_threshold, delta_threshold, display_info)
        character_sim, global_sim = None, None
        if display_info:
            character_sim = [[] for _ in range(components)]
            global_sim = [None for _ in range(components)]
            idx_offset = 0
            for component_idx, component in enumerate(component_list):
                for code in known_code:
                    name = self.speaker_label.get_name_by_code(code)
                    overall_score = active_idx_count[component_idx][code] / component.get_size() + max_max[component_idx][code] + \
                                    max_median[component_idx][code] + median_max[component_idx][code] + mean_avg_top[component_idx][code]
                    character_sim[component_idx].append({
                        "name": name,
                        "active_idx_count": active_idx_count[idx][code],
                        "max_max": max_max[component_idx][code],
                        "max_median": max_median[component_idx][code],
                        "median_max": median_max[component_idx][code],
                        "mean_avg_top": mean_avg_top[component_idx][code],
                        "overall_score": overall_score,
                        "is_high_similarity": is_high_similarity[component_idx][code]
                    })
                    global_sim[component_idx] = global_sim_arr[idx_offset: idx_offset + idx_list_length[component_idx], :]
        highest_similarity_character = [] * components
        highest_similarity = [] * components
        high_similarity_count = [0] * components
        if compute_info:
            for component_idx, component in enumerate(component_list):
                character_list, similarity_list = [], []
                for code in known_code:
                    character_name = curr_label_set.get_name_by_code(code)
                    if active_idx_count[component_idx][code] > 0 or is_high_similarity[component_idx][code]:
                        if is_high_similarity[component_idx][code]:
                            high_similarity_count[component_idx] += 1
                        character_list.append(character_name)
                        similarity_list.append(mean_avg_top[component_idx][code])
                sorted_character_idx = sorted(range(len(similarity_list)), key=lambda x: -similarity_list[x])
                highest_similarity_character.append([character_list[x] for x in sorted_character_idx])
                highest_similarity.append([similarity_list[x] for x in sorted_character_idx])
        return character_sim, global_sim, highest_similarity_character, highest_similarity, high_similarity_count

    def _set_high_similarity_characters(self, info_set, curr_label_set, similarity, component_list,
                                        sim_threshold_base, sim_threshold_new, face_radius, face_score_threshold):
        _, _, highest_similarity_character, highest_similarity, high_similarity_count = \
            self._get_similarity_information(info_set, curr_label_set, similarity, component_list,
                                             sim_threshold_base, sim_threshold_new, face_radius, face_score_threshold, compute_info=True)
        for component_idx, component in enumerate(component_list):
            if len(highest_similarity[component_idx]) > 0 and component.get_character_name() is None:
                if high_similarity_count[component_idx] > 0:
                    component.set_character_name(highest_similarity_character[component_idx][0])
                component.set_similarity(highest_similarity[component_idx][0])
                component.set_is_uncertain(high_similarity_count[component_idx] != 1)

    def _detect_local_components(self, info_set, curr_label_set, similarity,
                                 sim_threshold_base, sim_threshold_new, size_threshold, subtitle_radius, face_radius, face_score_threshold):
        active_idx = np.array(curr_label_set.get_unlabeled_utterance_idx(), dtype=np.int32)
        local_cc_list = self._get_components_by_similarity(similarity, active_idx, None,
                                                           sim_threshold_base + sim_threshold_new, size_threshold, subtitle_radius)
        self._set_high_similarity_characters(info_set, curr_label_set, similarity, local_cc_list,
                                             sim_threshold_base, sim_threshold_new, face_radius, face_score_threshold)
        return local_cc_list

    def propagate_with_threshold(self, info_set, curr_label_set, similarity,
                                 sim_threshold_base, sim_threshold_new, size_threshold, subtitle_radius, face_radius, face_score_threshold):
        new_label_item_list = self._get_propagated_label_set(info_set, curr_label_set, similarity,
                                                             sim_threshold_base, face_radius, face_score_threshold)
        local_cc_list = self._detect_local_components(info_set, curr_label_set, similarity,
                                                      sim_threshold_base, sim_threshold_new, size_threshold, subtitle_radius, face_radius, face_score_threshold)
        return new_label_item_list, local_cc_list
