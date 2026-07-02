import os
import sys
import time
import numpy as np
import json
from enum import Enum

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from data_loader import DramaData
from graph_algorithm import GraphAlgorithm
from label_structure import LabelSet


class LabelPropagation:

    class InitSeedsOption(Enum):
        NONE = "none"
        ONE_PERCENT = "1pct"
        ONE_SHOT = "1shot"
        def __str__(self):
            return self.value

    class LabelPropOption(Enum):
        NONE = "none"
        AP_WIN = "apw"
        def __str__(self):
            return self.value

    class Parameters:
        SIM_THRES_HIGH_SI = 0.85
        SIM_THRES_MEDIUM_SI = 0.75
        SIM_THRES_LOW_SI = 0.70
        SIM_THRES_STEP_SI = 0.01
        SIZE_THRES_HIGH_SI = 100
        SIZE_THRES_MEDIUM_SI = 50
        SIZE_THRES_LOW_SI = 10
        SIM_THRES_HIGH_LP = 0.75
        SIM_THRES_LOW_LP = 0.45
        SIM_THRES_STEP_LP = 0.02
        SIM_THRES_NEW_LP = 0.08
        SIM_THRES_EPSILON = 1e-6
        FACE_RADIUS_LP = 20000
        FACE_SCORE_THRES_LP = 0.45
        SIZE_THRES_LP = 3
        SEARCH_RADIUS_NEW_LP = 5
        TOP_UTTERANCES = lambda total_utterances: max(1, min(50, int(total_utterances ** 0.4)))

    def __init__(self, project, args):
        self.project = project
        self.args = args
        self.drama_name = args.drama_name
        self.drama_data = project.drama_data
        self.initial_character_info = self.drama_data.get_character_info()
        self.label_set = LabelSet(self.drama_data).reset(list(self.initial_character_info.keys()), OFFSCREEN_KEYWORD_LIST(self.drama_data.get_language()))
        self.info_set = self._get_info_set()
        self.algorithm = GraphAlgorithm()
        self.similarity = None

    def _get_info_set(self):
        assert self.drama_data is not None, "Error: drama data shall not be empty."
        subtitle_data, face_data = self.drama_data.subtitle_data, self.drama_data.face_data
        subtitle_timestamp = [[0, 0] for _ in range(subtitle_data["subtitles"])]
        for subtitle_idx in range(subtitle_data["subtitles"]):
            subtitle_timestamp[subtitle_idx][0] = subtitle_data["subtitle_list"][subtitle_idx]["abs_start_timestamp"]
            subtitle_timestamp[subtitle_idx][1] = subtitle_data["subtitle_list"][subtitle_idx]["abs_end_timestamp"]
        face_timestamp = [[0, 0] for _ in range(face_data["faces"])]
        face_code = [-1] * face_data["faces"]
        face_score = [0.] * face_data["faces"]
        for face_idx in range(face_data["faces"]):
            face_timestamp[face_idx][0] = face_data["face_list"][face_idx]["abs_start_timestamp"]
            face_timestamp[face_idx][1] = face_data["face_list"][face_idx]["abs_end_timestamp"]
            character = face_data["face_list"][face_idx]["character"]
            if character in self.label_set.character_dict:
                face_code[face_idx] = self.label_set.character_dict[character]
            face_score[face_idx] = face_data["face_list"][face_idx]["score"]
        info_set = {
            "subtitle_timestamp": subtitle_timestamp,
            "face_timestamp": face_timestamp,
            "face_code": face_code,
            "face_score": face_score,
        }
        return info_set

    def get_target_utterances(self, init_seeds_type, total_utterances):
        if init_seeds_type == LabelPropagation.InitSeedsOption.NONE:
            return 0
        elif init_seeds_type == LabelPropagation.InitSeedsOption.ONE_SHOT:
            return 1
        elif init_seeds_type == LabelPropagation.InitSeedsOption.ONE_PERCENT:
            return max(int(total_utterances * 0.01), 1)
        else:
            assert False, "Unknown InitSeedsOption argument."

    def generate_init_seeds(self):
        label_file = os.path.join(self.args.drama_dir, INIT_SEEDS_FILE(self.args.init_seeds_str))
        if not self.args.enforce_refresh and os.path.exists(label_file):
            print_with_indent("Init seeds file existed; skipped.", indent=INDENT_INFO)
        else:
            if self.similarity is None:
                self.similarity = self.algorithm.compute_similarity(self.drama_data.sv_embedding["embedding"])
            start_time = time.time()
            print_with_indent("Starting generating init seeds with option <{:s}>...".format(str(self.args.init_seeds_option)), indent=INDENT_INFO)
            if self.args.init_seeds_option == LabelPropagation.InitSeedsOption.NONE:
                assert False, "InitSeedsOption.NONE is not supported yet!"
            else:
                initial_character_list = list(self.initial_character_info.keys())
                self.label_set.reset(initial_character_list, OFFSCREEN_KEYWORD_LIST(self.drama_data.get_language()))
                characters, seed_utterances = 0, 0
                for character in initial_character_list:
                    subtitle_idx_list = self.initial_character_info[character]
                    total_utterances = len(subtitle_idx_list)
                    target_utterances = self.get_target_utterances(self.args.init_seeds_option, total_utterances)
                    print_with_indent("Character {:s}: extracting {:d} of {:d} utterances as init seeds...".format(
                        character, target_utterances, total_utterances), indent=INDENT_INFO, ending=False)
                    confident_cc = self.algorithm.get_most_confident_component(self.similarity, subtitle_idx_list, target_utterances)
                    self.label_set.label_idx_list(character, confident_cc, None, is_gt=True)
                    real_utterances = len(confident_cc)
                    characters += 1
                    seed_utterances += real_utterances
                    print_with_indent("Done with {:d} utterances; {:0.6f} seconds elapsed.".format(
                        real_utterances, time.time() - start_time), indent=INDENT_INFO, ending=True)
            self.label_set.save_to(label_file)
            print_with_indent("Finished generating init seeds for {:d} characters and {:d} utterances; {:0.6f} seconds elapsed.".format(
                characters, seed_utterances, time.time() - start_time), indent=INDENT_INFO)

    def load_init_seeds(self):
        print_with_indent("Starting loading init seeds with option <{:s}>...".format(str(self.args.init_seeds_option)), indent=INDENT_INFO)
        label_file = os.path.join(self.args.drama_dir, INIT_SEEDS_FILE(self.args.init_seeds_str))
        self.label_set.load_from(label_file)
        characters = self.label_set.characters
        seed_utterances = sum([len(self.label_set.character_list[character_idx]["utterance_list"]) for character_idx in range(characters)])
        print_with_indent("Finished loading init seeds with option <{:s}>, {:d} characters and {:d} utterances have been labeled.".format(
            str(self.args.init_seeds_option), characters, seed_utterances), indent=INDENT_INFO)

    def accept_label_item_list(self, label_item_list):
        for label_item in label_item_list:
            idx, character_name = label_item.get_idx(), label_item.get_likely_name()
            assert character_name is not None and self.label_set.is_existed_name(character_name), "Error: invalid character name {:s}".format(character_name)
            self.label_set.label_idx_list(label_item.get_likely_name(), [label_item.get_idx()], [label_item.get_rel_info()])

    def label_local_components(self, local_cc_list, as_others=False, accept_uncertain=False, is_gt=False):
        character_labeled = False
        if not as_others:
            for component in local_cc_list:
                character_name = component.get_character_name()
                if character_name is not None and (accept_uncertain or not component.get_is_uncertain()):
                    score = component.get_similarity()
                    rel_info_list = [{"_list": [character_name], "_dict": {character_name: score}} for _ in range(component.get_size())]
                    self.label_set.label_idx_list(component.get_character_name(), component.get_idx_list(), rel_info_list)
                    character_labeled = True
                    print_with_indent(">> A local component attributed to character {:s}.".format(component.get_character_name()), indent=INDENT_LOG)
        else:
            for component in local_cc_list:
                if component.get_character_name() is None:
                    self.label_set.label_idx_list_special(LabelSet.SpecialCode.CODE_OTHERS, component.get_idx_list())
                    character_labeled = True
        return character_labeled

    @staticmethod
    def _get_average(sub_similarity):
        total_utterances = sub_similarity.shape[1]
        top_utterances = LabelPropagation.Parameters.TOP_UTTERANCES(total_utterances)
        # NOTE: np.partition() is even slower than np.sort()
        return np.mean(np.sort(sub_similarity, axis=1)[:, -top_utterances:], axis=1)

    def finalize_rel_info(self):
        parameters = LabelPropagation.Parameters
        active_map = self.algorithm._get_active_map(self.info_set, self.label_set, parameters.FACE_RADIUS_LP, parameters.FACE_SCORE_THRES_LP)
        subtitles, characters = active_map.shape[0], active_map.shape[1]
        score_arr = np.zeros_like(active_map, dtype=np.float32)
        for character_name in self.label_set.get_character_name_list():
            print_with_indent("Finalizing rel_info for character {:s}...".format(character_name), indent=INDENT_LOG)
            code = self.label_set.get_code_by_name(character_name)
            labeled_utterance_idx = self.label_set.get_utterance_idx_by_code(code)
            if len(labeled_utterance_idx) == 0:
                continue
            active_utterance_idx = np.where(active_map[:, code] > 0)[0]
            if len(active_utterance_idx) == 0:
                continue
            score_arr[active_utterance_idx, code] = LabelPropagation._get_average(self.similarity[active_utterance_idx, :][:, labeled_utterance_idx])
        for subtitle_idx in range(subtitles):
            active_code = np.argsort(-score_arr[subtitle_idx, :])
            rel_info_list, rel_info_dict = [], {}
            for character_idx in range(characters):
                if score_arr[subtitle_idx, active_code[character_idx]] > 0:
                    character_name = self.label_set.get_name_by_code(active_code[character_idx])
                    score = float(score_arr[subtitle_idx, active_code[character_idx]])
                    rel_info_list.append(character_name)
                    rel_info_dict.update({character_name: score})
            self.label_set.set_rel_info(subtitle_idx, {"_list": rel_info_list, "_dict": rel_info_dict})

    def propagate_affinities(self):
        label_file = os.path.join(self.args.drama_dir, PROP_LABELS_FILE(self.args.label_prop_str))
        if not self.args.enforce_refresh and os.path.exists(label_file):
            print_with_indent("Loading affinity propagation results...", indent=INDENT_INFO)
            self.label_set.load_from(label_file)
            print_with_indent("Finished loading affinity propagation results.", indent=INDENT_INFO)
        else:
            if self.similarity is None:
                self.similarity = self.algorithm.compute_similarity(self.drama_data.sv_embedding["embedding"])
            start_time = time.time()
            if self.args.label_prop_option == LabelPropagation.LabelPropOption.NONE:
                assert False, "LabelPropagation.LabelPropOption.NONE is not supported yet!"
            else:
                print_with_indent("Starting affinity propagation...", indent=INDENT_INFO)
                parameters = LabelPropagation.Parameters
                sim_threshold_curr, sim_threshold_new = parameters.SIM_THRES_HIGH_LP, parameters.SIM_THRES_NEW_LP
                total_utterances = self.label_set.get_total_utterances()
                while True:
                    labeled_utterances = self.label_set.get_labeled_utterances()
                    if labeled_utterances == total_utterances:
                        break
                    print_with_indent("Processing at threshold {:0.2f}, {:d}/{:d} utterances labeled.".format(
                        sim_threshold_curr, labeled_utterances, total_utterances), indent=INDENT_LOG)
                    new_label_item_list, local_cc_list = self.algorithm.propagate_with_threshold(
                        self.info_set, self.label_set, self.similarity, sim_threshold_curr, parameters.SIM_THRES_NEW_LP,
                        parameters.SIZE_THRES_LP, parameters.SEARCH_RADIUS_NEW_LP, parameters.FACE_RADIUS_LP, parameters.FACE_SCORE_THRES_LP)
                    current_threshold_finished = True
                    if new_label_item_list:
                        print_with_indent(">> Propagated {:d} utterances...".format(len(new_label_item_list)), indent=INDENT_LOG)
                        self.accept_label_item_list(new_label_item_list)
                    if local_cc_list:
                        print_with_indent(">> Plausible new character detected with {:d} utterances...".format(len(local_cc_list)), indent=INDENT_LOG)
                        self.label_local_components(local_cc_list)
                        for component in local_cc_list:
                            if component.get_character_name() is None and component.get_similarity() + sim_threshold_new < sim_threshold_curr:
                                new_character_name = self.label_set.get_next_available_name(NEW_CHARACTER_NAME_FUNC[self.drama_data.get_language()])
                                rel_info_list = [{"_list": [new_character_name], "_dict": {new_character_name: 1.}} for _ in range(component.get_size())]
                                self.label_set.label_idx_list(new_character_name, component.get_idx_list(), rel_info_list)
                                current_threshold_finished = False
                    if current_threshold_finished:
                        sim_threshold_curr -= parameters.SIM_THRES_STEP_LP
                        if sim_threshold_curr <= parameters.SIM_THRES_LOW_LP - parameters.SIM_THRES_EPSILON:
                            break
                self.label_set.label_remaining_as_others()
                self.finalize_rel_info()
                self.label_set.save_to(label_file)
                print_with_indent("Finished affinity propagation; {:0.6f} seconds elapsed.".format(
                    time.time() - start_time), indent=INDENT_INFO)

    def compute_accuracy(self):
        print_with_indent("Speaker recognition accuracy of label propagation:", indent=INDENT_INFO)
        corrects_by_episode, utterances_by_episode = self.drama_data.compute_prediction_accuracy(self.label_set.wrap_up_data())
        for episode_idx in range(self.drama_data.get_episodes()):
            print_with_indent("Episode {:04d}: {:d} / {:d} = {:0.2f}%".format(
                episode_idx + 1, corrects_by_episode[episode_idx], utterances_by_episode[episode_idx],
                100. * corrects_by_episode[episode_idx] / utterances_by_episode[episode_idx]), indent=INDENT_LOG)
        total_corrects, total_utterances = sum(corrects_by_episode), sum(utterances_by_episode)
        print_with_indent("Overall: {:d} / {:d} = {:0.2f}%".format(
            total_corrects, total_utterances, 100. * total_corrects / total_utterances), indent=INDENT_LOG)

    def perform_label_propagation(self):
        print_module_title("Starting label propagation...", is_start=True)
        self.load_init_seeds()
        self.propagate_affinities()
        self.compute_accuracy()
        print_module_title("Finished label propagation.", is_start=False)
