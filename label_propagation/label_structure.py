import os
import sys
import numpy as np
import json
from enum import IntEnum

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import CHARACTER_NAME_OTHERS, CHARACTER_NAME_UNKNOWN
from data_loader import DramaData


class Component:

    def __init__(self, idx_list, character_name=None, similarity=0., is_uncertain=False):
        self.idx_list = idx_list
        self.character_name = character_name
        self.similarity = similarity
        self.is_uncertain = is_uncertain

    def __repr__(self):
        return f"<{self.idx_list} ({self.character_name}, {self.similarity:.4f})>"

    def set_character_name(self, character_name):
        self.character_name = character_name
        return self

    def get_character_name(self):
        return self.character_name

    def set_idx_list(self, idx_list):
        self.idx_list = idx_list
        return self

    def get_idx_list(self):
        return self.idx_list

    def get_size(self):
        return len(self.idx_list)

    def set_similarity(self, similarity):
        self.similarity = similarity
        return self

    def get_similarity(self):
        return self.similarity

    def set_is_uncertain(self, is_uncertain):
        self.is_uncertain = is_uncertain

    def get_is_uncertain(self):
        return self.is_uncertain


class LabelItem:

    def __init__(self, idx, is_gt=False, rel_info=None):
        self.idx = idx
        self.is_gt = is_gt
        self.rel_info = {"_list": [], "_dict": {}} if rel_info is None else rel_info

    def __repr__(self):
        return f"<{self.idx}, {self.rel_info}>"

    def assign_rel_info(self, name_arr, sim_avg_arr):
        assert len(name_arr) == len(sim_avg_arr)
        rel_info_list = [name_arr[idx] for idx in range(len(name_arr))]
        rel_info_dict = {name_arr[idx]: sim_avg_arr[idx] for idx in range(len(name_arr))}
        self.rel_info = {"_list": rel_info_list, "_dict": rel_info_dict}
        return self

    def update_rel_info(self, name, sim_avg):
        left, right, target = 0, len(self.rel_info["_list"]) - 1, 0
        while left <= right:
            mid = (left + right) // 2
            if self._get_kth_score(mid) > sim_avg:
                target, left = mid + 1, mid + 1
            else:
                right = mid - 1
        self.rel_info["_list"].insert(target, name)
        self.rel_info["_dict"].update({name: sim_avg})
        return self

    def get_idx(self):
        return self.idx

    def get_rel_info(self):
        return self.rel_info

    def get_rel_info_size(self):
        return len(self.rel_info)

    def get_likely_name(self):
        return self.rel_info["_list"][0] if self.get_rel_info_size() > 0 else None

    def _get_kth_score(self, k):
        return self.rel_info["_dict"][self.rel_info["_list"][k]] if self.get_rel_info_size() >= k else None


class LabelSet:

    class SpecialCode(IntEnum):
        CODE_UNLABELED = -1
        CODE_DISABLED = -2
        CODE_OTHERS = -3
        CODE_UNKNOWN = -4
    SPECIAL_CHARACTER_NAME_DICT = {
        SpecialCode.CODE_OTHERS: CHARACTER_NAME_OTHERS,
        SpecialCode.CODE_UNKNOWN: CHARACTER_NAME_UNKNOWN,
    }

    def __init__(self, drama_data: DramaData):
        assert drama_data is not None, "Error: trying to load an empty DramaData object."
        self.drama_data = drama_data
        self.special_character_name_dict = {
            special_code: LabelSet.SPECIAL_CHARACTER_NAME_DICT[special_code][drama_data.get_language()]
            for special_code in LabelSet.SPECIAL_CHARACTER_NAME_DICT}

    def reset(self, known_character_list, offscreen_keyword_list, fill_value=True):
        self.known_character_list = known_character_list
        self.subtitle_data = self.drama_data.subtitle_data
        self.utterances = self.subtitle_data["subtitles"]
        if fill_value:
            self.characters = len(known_character_list)
            self.character_list = [{
                "name": known_character_list[character_idx],
                "code": character_idx,
                "is_known": True,
                "is_global": any([offscreen_keyword_list[idx] in known_character_list[character_idx] for idx in range(len(offscreen_keyword_list))]),
                "utterance_list": []
            } for character_idx in range(self.characters)]
            self.character_dict = {known_character_list[character_idx]: character_idx for character_idx in range(self.characters)}
            self.utterance_label = [{
                "code": -1,
                "is_gt": False,
                "rel_info": []
            } for utterance_idx in range(self.utterances)]
            self.code_arr = np.ones(self.utterances, dtype=np.int32) * (-1)
        return self

    def load_from(self, label_file):
        loaded_data = json.load(open(label_file, "r", encoding="utf-8"))
        self.reset(list(loaded_data["character_dict"].keys()), [], fill_value=False)
        self.characters = loaded_data["characters"]
        self.character_list = loaded_data["character_list"]
        self.character_dict = loaded_data["character_dict"]
        self.utterance_label = loaded_data["utterance_label"]
        self.code_arr = np.array([self.utterance_label[utterance_idx]["code"] for utterance_idx in range(self.utterances)], dtype=np.int32)
        assert self.utterances == len(self.utterance_label), "Error: the number of loaded utterances does not match that of drama data."
        return self

    def save_to(self, label_file):
        saved_data = {
            "characters": self.characters,
            "character_list": self.character_list,
            "character_dict": self.character_dict,
            "utterance_label": self.utterance_label
        }
        json.dump(saved_data, open(label_file, "w", encoding="utf-8"), ensure_ascii=False)
        return self

    def wrap_up_data(self):
        utterances = self.get_total_utterances()
        wrapped_data = {
            "utterances": utterances,
            "utterance_list": [{"character_name": self.get_name_by_code(self.code_arr[utterance_idx])} for utterance_idx in range(utterances)]
        }
        return wrapped_data

    def _label_idx_list_with_code(self, code, utterance_idx_list, rel_info_list, is_gt=False):
        if code >= 0:
            self.character_list[code]["utterance_list"].extend(utterance_idx_list)
        if rel_info_list is None:
            assert is_gt, "Error: cannot accept non-GT labels with empty rel_info arrays."
            character_name = self.get_name_by_code(code)
            rel_info_list = [{"_list": [character_name], "_dict": {character_name: 1.}} for _ in range(len(utterance_idx_list))]
        for _, (utterance_idx, rel_info) in enumerate(zip(utterance_idx_list, rel_info_list)):
            self.utterance_label[utterance_idx] = {
                "code": code,
                "is_gt": is_gt,
                "rel_info": rel_info
            }
            self.code_arr[utterance_idx] = code
        return self

    def label_idx_list(self, character_name, utterance_idx_list, rel_info_list, is_gt=False):
        if not character_name in self.character_dict:
            self.character_list.append({
                "name": character_name,
                "code": self.characters,
                "is_known": False,
                "is_global": False,
                "utterance_list": []
            })
            self.character_dict.update({character_name: self.characters})
            self.characters += 1
        code = self.character_dict[character_name]
        self._label_idx_list_with_code(code, utterance_idx_list, rel_info_list, is_gt=is_gt)
        return self

    def label_idx_list_special(self, code, utterance_idx_list, is_gt=False):
        assert code in LabelSet.SPECIAL_CHARACTER_NAME_DICT, "Error: assigned code {:d} is not a special code.".format(code)
        if len(utterance_idx_list) > 0:
            self._label_idx_list_with_code(code, utterance_idx_list, [{"_list": [], "_dict": {}} for _ in range(len(utterance_idx_list))], is_gt=is_gt)

    def label_remaining_as_others(self):
        utterance_idx_list = self.get_unlabeled_utterance_idx()
        self.label_idx_list_special(LabelSet.SpecialCode.CODE_OTHERS, utterance_idx_list)

    def set_rel_info(self, utterance_idx, rel_info):
        self.utterance_label[utterance_idx].update({"rel_info": rel_info})

    def get_prediction_info(self, utterance_idx):
        name = self.get_name_by_code(self.utterance_label[utterance_idx]["code"])
        # WARNING: this definition is temporary and ad-hoc
        rel_info = self.utterance_label[utterance_idx]["rel_info"]
        candidates = len(rel_info["_list"])
        sim_top1, sim_advantage = 0., 0.
        if candidates == 1:
            sim_top1 = sim_advantage = rel_info["_dict"][rel_info["_list"][0]]
            if sim_advantage < 1.:
                sim_advantage -= 0.25
        elif candidates > 1:
            sim_top1 = rel_info["_dict"][rel_info["_list"][0]]
            sim_advantage = sim_top1 - rel_info["_dict"][rel_info["_list"][1]]
        return name, sim_top1, sim_advantage

    def get_known_characters(self):
        return self.characters

    def get_character_name_list(self):
        return [self.character_list[character_idx]["name"] for character_idx in range(self.characters)]

    def get_total_utterances(self):
        return self.utterances

    def get_labeled_utterances(self):
        return sum([len(self.character_list[character_idx]["utterance_list"]) for character_idx in range(self.characters)])

    def get_utterance_idx_by_code(self, code):
        return np.where(self.code_arr == code)[0].astype(np.int32)

    def get_unlabeled_utterance_idx(self):
        return np.where(np.logical_or(self.code_arr == LabelSet.SpecialCode.CODE_UNLABELED,
                                      self.code_arr == LabelSet.SpecialCode.CODE_DISABLED))[0].astype(np.int32).tolist()

    def is_existed_name(self, character_name):
        return character_name in self.character_dict

    def get_next_available_name(self, name_func):
        min_idx, max_idx = 1, 1
        while True:
            if not self.is_existed_name(name_func(max_idx)):
                break
            min_idx, max_idx = max_idx, max_idx * 2
        while max_idx > min_idx:
            mid_idx = (min_idx + max_idx) // 2
            if self.is_existed_name(name_func(mid_idx)):
                min_idx = mid_idx + 1
            else:
                max_idx = mid_idx
        return name_func(max_idx)

    def get_code_by_name(self, character_name):
        return self.character_dict[character_name]

    def get_name_by_code(self, code):
        return self.special_character_name_dict[code] if code in LabelSet.SPECIAL_CHARACTER_NAME_DICT else self.character_list[code]["name"]

    def get_global_code_list(self):
        return [self.character_list[character_idx]["code"] for character_idx in range(self.characters) if self.character_list[character_idx]["is_global"]]

    def get_mask_by_component_list(self, component_list):
        mask = np.zeros(self.utterances, dtype=np.uint8)
        for component in component_list:
            mask[component.get_idx_list()] = 1
        return mask
