import os
import time
import numpy as np
import json
import re
from enum import Enum

from utils import OFFSCREEN_KEYWORD_LIST, is_others_character, is_openset_character, is_invalid_character
from utils import print_module_title, print_with_indent, INDENT_MODULE, INDENT_INFO, INDENT_LOG
from utils import time_str2int


class DramaData:

    SUBTITLE_DATA_DIR = "subtitle_data"
    FACE_DATA_DIR = "face_data"
    FACE_INFO_FILE = "face_info.json"
    SV_EMBEDDING_DIR = "sv_embedding"
    GT_SPEAKER_DATA_DIR = "gt_speaker"
    CAPTION_DATA_FILE = "caption.json"
    RELATION_DATA_FILE = "relation.json"

    EPISODE_PADDING = 10000
    UNKNOWN_FACE_PATTERN = "[0-9]{4}\_pro[0-9]+\_id[0-9]+"

    def __init__(self, project, args):
        self.project = project
        self.args = args
        self.is_loaded = False
        self.drama_name = args.drama_name
        self.episode_info = {
            "language": self.args.language,
            "episodes": self.args.episodes
        }
        self.subtitle_data = None
        self.face_data = None
        self.known_character_list = None
        self.sv_embedding = None
        self.caption_data = None
        self.relation_data = None
        self.init_seeds = None

    def load_subtitle_data(self):
        start_time = time.time()
        print_with_indent("Starting loading subtitle data...", indent=INDENT_INFO)
        subtitle_data_dir = os.path.join(self.args.drama_dir, DramaData.SUBTITLE_DATA_DIR)
        self.subtitle_data = {"episodes": self.args.episodes}
        mainbody_list = []
        subtitle_list = []
        total_duration, object_idx_offset = 0, 0
        for episode_idx in range(self.args.episodes):
            subtitle_file_episode = os.path.join(subtitle_data_dir, "{:04d}.json".format(episode_idx + 1))
            subtitle_data_episode = json.load(open(subtitle_file_episode, "r", encoding="utf-8"))
            start_timestamp, end_timestamp = -1, -1
            if "mainbody" in subtitle_data_episode:
                mainbody = subtitle_data_episode["mainbody"]
                if "start_time" in mainbody:
                    start_timestamp = time_str2int(mainbody["start_time"])
                if "end_time" in mainbody:
                    end_timestamp = time_str2int(mainbody["end_time"])
            if start_timestamp == -1:
                subtitle_object = subtitle_data_episode["object_list"][str(0)]
                episode_start_timestamp = time_str2int(subtitle_object["start_time"]) - DramaData.EPISODE_PADDING
            if end_timestamp == -1:
                subtitle_object = subtitle_data_episode["object_list"][str(subtitle_data_episode["objects"] - 1)]
                episode_end_timestamp = time_str2int(subtitle_object["end_time"]) + DramaData.EPISODE_PADDING
            mainbody_list.append({"start_timestamp": episode_start_timestamp, "end_timestamp": episode_end_timestamp})
            for object_idx in range(subtitle_data_episode["objects"]):
                subtitle_object = subtitle_data_episode["object_list"][str(object_idx)]
                subtitle_list.append({
                    "episode_idx": int(subtitle_object["episode_idx"]),
                    "object_idx": int(subtitle_object["object_idx"]),
                    "overall_idx": int(subtitle_object["object_idx"]) + object_idx_offset,
                    "start_timestamp": time_str2int(subtitle_object["start_time"]),
                    "end_timestamp": time_str2int(subtitle_object["end_time"]),
                    "abs_start_timestamp": time_str2int(subtitle_object["start_time"]) - episode_start_timestamp + total_duration,
                    "abs_end_timestamp": time_str2int(subtitle_object["end_time"]) - episode_start_timestamp + total_duration,
                    "text": subtitle_object["text"],
                    "character_gt": subtitle_object["character"] if "character" in subtitle_object else None,
                    "is_invalid": is_invalid_character(subtitle_object["character"]) if "character" in subtitle_object else False,
                })
            total_duration += (episode_end_timestamp - episode_start_timestamp)
            object_idx_offset += subtitle_data_episode["objects"]
        self.episode_info.update({"mainbody_list": mainbody_list})
        self.subtitle_data.update({"subtitles": len(subtitle_list), "subtitle_list": subtitle_list})
        elapsed_time = time.time() - start_time
        print_with_indent("{:d} subtitle objects loaded; {:0.6f} seconds elapsed.".format(
            self.subtitle_data["subtitles"], elapsed_time), indent=INDENT_LOG)
        print_with_indent("Finished loading subtitle data.", indent=INDENT_INFO)

    def load_face_data(self):
        start_time = time.time()
        print_with_indent("Starting loading face data...", indent=INDENT_INFO)
        face_data_dir = os.path.join(self.args.drama_dir, DramaData.FACE_DATA_DIR)
        face_info_file = os.path.join(face_data_dir, DramaData.FACE_INFO_FILE)
        face_info = json.load(open(face_info_file, "r", encoding="utf-8"))
        self.known_character_list = list(face_info["role_name_variants"].keys())
        known_character_appeared = [False] * len(self.known_character_list)
        self.face_data = {"episodes": self.args.episodes}
        face_list = []
        total_duration, object_idx_offset = 0, 0
        for episode_idx in range(self.args.episodes):
            episode_start_timestamp = self.episode_info["mainbody_list"][episode_idx]["start_timestamp"]
            episode_end_timestamp = self.episode_info["mainbody_list"][episode_idx]["end_timestamp"]
            face_file_episode = os.path.join(face_data_dir, "{:04d}.json".format(episode_idx + 1))
            face_data_episode = json.load(open(face_file_episode, "r", encoding="utf-8"))
            for object_idx in range(face_data_episode["objects"]):
                face_object = face_data_episode["object_list"][str(object_idx)]
                character = face_object["id_list"][str(0)]["name"]
                score = float(face_object["id_list"][str(0)]["score"])
                face_list.append({
                    "episode_idx": int(face_object["episode_idx"]),
                    "object_idx": int(face_object["object_idx"]),
                    "overall_idx": int(face_object["object_idx"]) + object_idx_offset,
                    "start_timestamp": time_str2int(face_object["start_time"]),
                    "end_timestamp": time_str2int(face_object["end_time"]),
                    "abs_start_timestamp": time_str2int(face_object["start_time"]) - episode_start_timestamp + total_duration,
                    "abs_end_timestamp": time_str2int(face_object["end_time"]) - episode_start_timestamp + total_duration,
                    "character": character,
                    "score": score,
                    "is_known": character in self.known_character_list,
                })
                if character in self.known_character_list:
                    known_character_appeared[self.known_character_list.index(character)] = True
            total_duration += (episode_end_timestamp - episode_start_timestamp)
            object_idx_offset += face_data_episode["objects"]
        self.known_character_list = [self.known_character_list[idx] for idx in range(len(self.known_character_list)) if known_character_appeared[idx]]
        self.face_data.update({"faces": len(face_list), "face_list": face_list})
        elapsed_time = time.time() - start_time
        print_with_indent("{:d} face objects loaded; {:0.6f} seconds elapsed.".format(
            self.face_data["faces"], elapsed_time), indent=INDENT_LOG)
        print_with_indent("Finished loading face data.", indent=INDENT_INFO)

    def load_sv_embedding(self):
        start_time = time.time()
        print_with_indent("Starting loading SV embedding...", indent=INDENT_INFO)
        sv_embedding_dir = os.path.join(self.args.drama_dir, DramaData.SV_EMBEDDING_DIR)
        self.sv_embedding = {"episodes": self.args.episodes}
        embedding = None
        for episode_idx in range(self.args.episodes):
            sv_embedding_file_episode = os.path.join(sv_embedding_dir, "{:04d}.npy".format(episode_idx + 1))
            embedding_episode = np.load(sv_embedding_file_episode)
            if embedding is None:
                embedding = embedding_episode
            else:
                embedding = np.concatenate((embedding, embedding_episode), axis=0)
        assert (self.subtitle_data["subtitles"] == embedding.shape[0]), \
            "Error: the total number of subtitles and the SV embedding's 1st dimensionality do not match."
        self.sv_embedding.update({"embedding": embedding})
        elapsed_time = time.time() - start_time
        print_with_indent("SV embedding loaded with dimensionality {:d} x {:d}; {:0.6f} seconds elapsed.".format(
            embedding.shape[0], embedding.shape[1], elapsed_time), indent=INDENT_LOG)
        print_with_indent("Finished loading SV embedding.", indent=INDENT_INFO)

    def load_caption_data(self):
        start_time = time.time()
        print_with_indent("Starting loading caption data...", indent=INDENT_INFO)
        caption_data_file = os.path.join(self.args.drama_dir, DramaData.CAPTION_DATA_FILE)
        caption_data = json.load(open(caption_data_file, "r", encoding="utf-8"))
        brief_caption_list, detailed_caption_list = [], []
        episode_idx, total_duration = 0, 0
        episode_start_timestamp = self.episode_info["mainbody_list"][0]["start_timestamp"]
        episode_end_timestamp = self.episode_info["mainbody_list"][0]["end_timestamp"]
        for brief_caption_idx in range(len(caption_data)):
            brief_caption_object = caption_data[brief_caption_idx]
            while episode_idx < brief_caption_object["episode_idx"]:
                total_duration += (episode_end_timestamp - episode_start_timestamp)
                episode_idx += 1
                episode_start_timestamp = self.episode_info["mainbody_list"][episode_idx]["start_timestamp"]
                episode_end_timestamp = self.episode_info["mainbody_list"][episode_idx]["end_timestamp"]
            brief_caption_list.append({
                "episode_idx": episode_idx,
                "start_timestamp": time_str2int(brief_caption_object["brief_caption"]["start_timestamp"]),
                "end_timestamp": time_str2int(brief_caption_object["brief_caption"]["end_timestamp"]),
                "abs_start_timestamp": time_str2int(brief_caption_object["brief_caption"]["start_timestamp"]) - episode_start_timestamp + total_duration,
                "abs_end_timestamp": time_str2int(brief_caption_object["brief_caption"]["end_timestamp"]) - episode_start_timestamp + total_duration,
                "description": brief_caption_object["brief_caption"]["description"],
            })
            for detailed_caption_idx in range(len(brief_caption_object["detailed_caption_list"])):
                detailed_caption_object = brief_caption_object["detailed_caption_list"][detailed_caption_idx]
                detailed_caption_list.append({
                    "episode_idx": episode_idx,
                    "start_timestamp": time_str2int(detailed_caption_object["start_timestamp"]),
                    "end_timestamp": time_str2int(detailed_caption_object["end_timestamp"]),
                    "abs_start_timestamp": time_str2int(detailed_caption_object["start_timestamp"]) - episode_start_timestamp + total_duration,
                    "abs_end_timestamp": time_str2int(detailed_caption_object["end_timestamp"]) - episode_start_timestamp + total_duration,
                    "description": detailed_caption_object["description"],
                })
        self.caption_data = {
            "brief_caption_list": brief_caption_list,
            "detailed_caption_list": detailed_caption_list
        }
        elapsed_time = time.time() - start_time
        print_with_indent("{:d} brief and {:d} detailed captions loaded; {:0.6f} seconds elapsed.".format(
            len(brief_caption_list), len(detailed_caption_list), elapsed_time), indent=INDENT_LOG)
        print_with_indent("Finished loading caption data.", indent=INDENT_INFO)

    def load_relation_data(self, ensure_relation_data):
        print_with_indent("Start loading relation data...", indent=INDENT_INFO)
        relation_data_file = os.path.join(self.args.drama_dir, DramaData.RELATION_DATA_FILE)
        if os.path.exists(relation_data_file):
            start_time = time.time()
            self.relation_data = json.load(open(relation_data_file, "r", encoding="utf-8"))
            rolenames = len(self.relation_data["character_list"])
            relations = len(self.relation_data["relationship_list"])
            character_dict = {}
            for character in self.relation_data["character_list"]:
                character_dict.update({character: []})
                for relationship in self.relation_data["relationship_list"]:
                    if relationship[0] == character and relationship[1] in self.relation_data["character_list"]:
                        character_dict[character].append(relationship)
            self.relation_data.update({"character_dict": character_dict})
            elapsed_time = time.time() - start_time
            print_with_indent("Relation data loaded with {:d} characters and {:d} relationships; {:0.6f} seconds elapsed.".format(
                rolenames, relations, elapsed_time), indent=INDENT_LOG)
        else:
            assert not ensure_relation_data, "Error: relation data does not exist."
            print_with_indent("Relation data loading is skipped.", indent=INDENT_LOG)
        print_with_indent("Finished loading relation data.", indent=INDENT_INFO)

    def load_all_drama_data(self, ensure_relation_data=True, ensure_init_seeds=True):
        if not self.is_loaded:
            print_module_title("Starting loading all drama data...", is_start=True)
            self.drama_name = self.args.drama_name
            self.load_subtitle_data()
            self.load_face_data()
            self.load_sv_embedding()
            self.load_caption_data()
            self.load_relation_data(ensure_relation_data)
            print_module_title("Finished loading all drama data.", is_start=False)
            self.is_loaded = True
        return self

    def get_language(self):
        return self.episode_info["language"]

    def get_episodes(self):
        return self.episode_info["episodes"]

    def get_character_info(self):
        subtitle_data, known_character_list = self.subtitle_data, self.known_character_list
        offscreen_keyword_list = OFFSCREEN_KEYWORD_LIST(self.get_language())
        character_info = {}
        for subtitle_object in subtitle_data["subtitle_list"]:
            if subtitle_object["is_invalid"]:
                continue
            character = subtitle_object["character_gt"]
            if character not in known_character_list:
                if any([keyword in character for keyword in offscreen_keyword_list]):
                    known_character_list.append(character)
            if character in known_character_list:
                if character not in character_info:
                    character_info.update({character: []})
                character_info[character].append(subtitle_object["overall_idx"])
        character_list = list(character_info.keys())
        utterance_count = [len(character_info[character]) for character in character_list]
        sorted_idx = [idx[0] for idx in sorted(enumerate(utterance_count), key=lambda _: -_[1])]
        return {character_list[idx]: character_info[character_list[idx]] for idx in sorted_idx if utterance_count[idx]}

    def is_prediction_correct(self, character_gt, character_pred, candidate_list=None):
        if character_gt in self.known_character_list:
            if candidate_list is None:
                return character_gt == character_pred
            elif character_gt in candidate_list:
                return character_gt == character_pred
            else:
                return character_gt == character_pred or is_others_character(character_pred, self.get_language())
        else:
            if candidate_list is None:
                return character_pred not in self.known_character_list
            elif character_gt in candidate_list:
                return character_gt == character_pred
            else:
                return character_gt == character_pred or is_openset_character(character_pred, self.get_language())

    def compute_prediction_accuracy(self, prediction_data):
        assert self.subtitle_data["subtitles"] == prediction_data["utterances"], "Error: the numbers of subtitles and utterances do not match!"
        corrects_by_episode, utterances_by_episode = [0] * self.get_episodes(), [0] * self.get_episodes()
        for subtitle_idx in range(self.subtitle_data["subtitles"]):
            subtitle_object = self.subtitle_data["subtitle_list"][subtitle_idx]
            if subtitle_object["is_invalid"]:
                continue
            episode_idx, character_gt = subtitle_object["episode_idx"], subtitle_object["character_gt"]
            character_pred = prediction_data["utterance_list"][subtitle_idx]["character_name"]
            corrects_by_episode[episode_idx] += self.is_prediction_correct(character_gt, character_pred)
            utterances_by_episode[episode_idx] += 1
        return corrects_by_episode, utterances_by_episode
