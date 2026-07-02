import os
import sys
import time
import numpy as np
import pandas as pd
import json
import re
import random
from enum import Enum
import threading
import concurrent.futures

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from label_propagation.label_structure import LabelSet
from toolset import Toolset
from api_caller import APICaller


class DataSampler:

    def __init__(self, sample_type_dict, sample_all=False):
        self.sampling_dict = {sample_type.value: {
            "sample_list": [],
            "existing_samples": 0,
            "prob": 1.0 if sample_all else min(prob, 1.0),
        } for sample_type, prob in sample_type_dict.items()}

    def add_new_sample_idx(self, sample_type, sample_idx, sample_file, exists):
        self.sampling_dict[sample_type.value]["sample_list"].append({
            "sample_idx": sample_idx,
            "sample_type": sample_type.value,
            "sample_file": sample_file,
            "exists": exists,
        })
        self.sampling_dict[sample_type.value]["existing_samples"] += exists

    def get_sample_list(self):
        sample_list = []
        for _, sample_info in self.sampling_dict.items():
            sample_list.extend([sample for __, sample in enumerate(sample_info["sample_list"]) if sample["exists"]])
            sample_pool = [sample for __, sample in enumerate(sample_info["sample_list"]) if not sample["exists"]]
            target_samples = max(1, int(len(sample_info["sample_list"]) * sample_info["prob"] + 0.5))
            sample_list.extend(random.sample(sample_pool, target_samples - sample_info["existing_samples"]))
        sample_list.sort(key=lambda item: item["sample_idx"])
        return sample_list


class DataCuration:

    class DataStage(Enum):
        SFT = "SFT trajectories"
        RL = "RL datapoints"
        INFER = "Model inference"
        def __str__(self):
            return self.value

    class SFTDataOption(Enum):
        NONE = "none"
        GEM3P = "gem3p"
        GPT5C = "gpt5c"
        # CLO45 = "clo45"
        def __str__(self):
            return self.value

    class RLDataOption(Enum):
        NONE = "none"
        def __str__(self):
            return self.value

    class SampleType(Enum):
        WRONG = "wrong"
        LOW_AND_CLOSE = "low_and_close"
        LOW_ONLY = "low_only"
        CLOSE_ONLY = "close_only"
        CONFIDENT = "confident"

    class Parameters:
        SIM_THRES_LOW = 0.45
        SIM_THRES_CLOSE = 0.03
        SAMPLING_RATIO_WRONG = 1.0
        SAMPLING_RATIO_LOW = 1.0
        SAMPLING_RATIO_CLOSE = 1.0
        SAMPLING_RATIO_CONFIDENT = 0.1
        CONTEXT_LENGTH = 30
        MAX_MESSAGE_LENGTH = 20
        MAX_RETRIES = 2

    def __init__(self, project, args):
        self.project = project
        self.args = args
        self.drama_data = project.drama_data
        self.drama_name = project.drama_data.drama_name
        self.subtitle_data = project.drama_data.subtitle_data
        self.label_set = LabelSet(self.drama_data)
        self.toolset = Toolset(args.language, self.drama_data, self.label_set)
        self.api_caller = None

    def load_pseudo_labels(self):
        initial_character_info = self.drama_data.get_character_info()
        self.label_set.reset(list(initial_character_info.keys()), OFFSCREEN_KEYWORD_LIST(self.drama_data.get_language()))
        prop_labels_file = os.path.join(self.args.drama_dir, PROP_LABELS_FILE(self.args.label_prop_str))
        self.label_set.load_from(prop_labels_file)

    def get_sample_data(self, data_in_dict, sample_file):
        if data_in_dict is not None:
            return data_in_dict
        elif os.path.exists(sample_file):
            return json.load(open(sample_file, "r", encoding="utf-8"))
        else:
            return None

    def is_sample_correct(self, subtitle_idx, sample_data):
        character_name = sample_data["answer"]
        if character_name is None:
            return False
        character_gt = self.subtitle_data["subtitle_list"][subtitle_idx]["character_gt"]
        return self.drama_data.is_prediction_correct(
            character_gt, character_name, candidate_list=self.toolset.get_candidate_str_list(subtitle_idx))

    def get_single_sft_trajectory(self, sample, data_in_dict, in_inference, gpu_id=None, max_retries=0):
        subtitle_idx, sample_type, sample_file = sample["sample_idx"], sample["sample_type"], sample["sample_file"]
        sample_data = self.get_sample_data(data_in_dict, sample_file)
        data_complete = sample_data is not None and self.is_sample_correct(subtitle_idx, sample_data)
        if sample_data is None or (self.args.enforce_retrial and not data_complete):
            parameters = DataCuration.Parameters
            context_length = parameters.CONTEXT_LENGTH
            max_message_length = parameters.MAX_MESSAGE_LENGTH
            max_retries = parameters.MAX_RETRIES if max_retries == 0 else max_retries
            character_gt = self.subtitle_data["subtitle_list"][subtitle_idx]["character_gt"]
            rel_info = self.label_set.utterance_label[subtitle_idx]["rel_info"]
            if not character_gt in rel_info["_list"]:
                character_gt = CHARACTER_NAME_OTHERS[self.drama_data.get_language()]
            api_failures, max_length_errors, incorrect_answers, retries = 0, 0, 0, 0
            message_list, is_complete, cheating_answer, final_answer = [], False, None, None
            while retries < max_retries:
                cheating_answer = None if incorrect_answers == 0 else character_gt
                message_list = self.toolset.get_initial_message(subtitle_idx, context_length, cheating_answer=cheating_answer)
                message_info = {"cheating": cheating_answer is not None, "tool_info_list": []}
                useful_message_mask = [True, True]
                while True:
                    api_success, content = self.api_caller.call_api(message_list, gpu_id=gpu_id)
                    if api_success:
                        finished, message_useful, tool_info, user_prompt = self.toolset.get_next_user_prompt(subtitle_idx, content, message_info)
                        message_list.append(self.toolset.get_assistant_message(content))
                        useful_message_mask.append(message_useful)
                        if finished:
                            if in_inference:
                                is_complete = True
                                final_answer = user_prompt
                            elif self.drama_data.is_prediction_correct(character_gt, user_prompt, candidate_list=self.toolset.get_candidate_str_list(subtitle_idx)):
                                is_complete = True
                                final_answer = user_prompt
                            else:
                                incorrect_answers += 1
                            break
                        else:
                            message_list.append(self.toolset.get_user_message(user_prompt))
                            useful_message_mask.append(message_useful)
                            if len(message_list) > max_message_length:
                                max_length_errors += 1
                                break
                    else:
                        if content == APICaller.APIFailureType.MAX_RETRIES_REACHED:
                            api_failures += 1
                            break
                        else:
                            assert False, "Error: unknown APIFailureType of API calls."
                if is_complete:
                    break
                retries += 1
            final_message_list = message_list
            if cheating_answer is not None:
                final_message_list = self.toolset.get_initial_message(subtitle_idx, context_length)
                final_message_list.extend(message_list[2:])
            sample_data = {
                "message_list": final_message_list,
                "useful_message_mask": useful_message_mask,
                "sample_type": sample_type,
                "caller_info": self.api_caller.get_info(),
                "cheating_answer": cheating_answer is not None,
                "is_complete": is_complete,
                "answer": final_answer,
            }
            json.dump(sample_data, open(sample_file, "w", encoding="utf-8"), ensure_ascii=False)
        return subtitle_idx, sample_data

    def get_single_rl_datapoint(self, sample, data_in_dict, in_inference, gpu_id=None):
        subtitle_idx, sample_type, sample_file = sample["sample_idx"], sample["sample_type"], sample["sample_file"]
        sample_data = self.get_sample_data(data_in_dict, sample_file)
        data_complete = sample_data is not None and self.is_sample_correct(subtitle_idx, sample_data)
        if sample_data is None or (self.args.enforce_retrial and not data_complete):
            parameters = DataCuration.Parameters
            context_length = parameters.CONTEXT_LENGTH
            init_message_list = self.toolset.get_initial_message(subtitle_idx, context_length)
            character_gt = self.subtitle_data["subtitle_list"][subtitle_idx]["character_gt"]
            sample_data = {
                "subtitle_idx": subtitle_idx,
                "candidate_list": self.toolset.get_candidate_str_list(subtitle_idx),
                "message_list": init_message_list,
                "sample_type": sample_type,
                "is_complete": True,
                "answer": character_gt,
            }
            json.dump(sample_data, open(sample_file, "w", encoding="utf-8"), ensure_ascii=False)
        return subtitle_idx, sample_data

    def get_data_config(self, data_stage):
        if data_stage == DataCuration.DataStage.SFT:
            data_sample_file = os.path.join(
                self.args.drama_data_dir,
                SFT_DATA_FILE(self.args.drama_name, self.args.sft_data_str))
            method_name = "get_single_sft_trajectory"
            sft_model_name = self.args.sft_data_option
            self.api_caller = APICaller(
                model_proprietary=sft_model_name.value,
                api_key=self.args.api_key,
                base_url=self.args.base_url)
            max_workers = self.args.sft_workers
        elif data_stage == DataCuration.DataStage.RL:
            data_sample_file = os.path.join(
                self.args.drama_data_dir,
                RL_DATA_FILE(self.drama_name, self.args.rl_data_str))
            method_name = "get_single_rl_datapoint"
            max_workers = self.args.rl_workers
        elif data_stage == DataCuration.DataStage.INFER:
            data_sample_file = os.path.join(
                self.args.test_result_dir,
                INFER_DATA_FILE(self.drama_name, self.args.infer_data_str))
            method_name = "get_single_sft_trajectory"
            infer_model_name = self.args.target_model
            self.api_caller = APICaller(
                model_local=infer_model_name.value,
                api_key=self.args.api_key_local,
                base_port_id=self.args.base_port_id,
                num_gpus=self.args.num_gpus,
                args=self.args)
            max_workers = self.args.infer_workers
        else:
            assert False, "Unsupported data stage!"
        data_sample_dir = os.path.splitext(data_sample_file)[0]
        in_inference = (data_stage == DataCuration.DataStage.INFER)
        return data_sample_file, data_sample_dir, method_name, max_workers, in_inference

    def collect_all_data(self, data_dict, sample_list, method_name, max_workers, parameters, in_inference):
        func = getattr(self, method_name)
        total_samples, complete_samples, correct_samples = 0, 0, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(func, sample, data_dict[str(sample["sample_idx"])],
                                             in_inference, sample_idx % self.args.num_gpus):
                             sample_idx for sample_idx, sample in enumerate(sample_list)}
            for future in concurrent.futures.as_completed(future_to_idx):
                subtitle_idx, sample_data = future.result()
                data_dict.update({str(subtitle_idx): sample_data})
                total_samples += 1
                if sample_data["is_complete"]:
                    complete_samples += 1
                    if self.is_sample_correct(subtitle_idx, sample_data):
                        correct_samples += 1
                    elif not in_inference:
                        print_with_indent(">> Warning: an incorrect sample, #{:d}.".format(subtitle_idx), indent=INDENT_LOG)
                else:
                    print_with_indent(">> Warning: an incomplete sample, #{:d}.".format(subtitle_idx), indent=INDENT_LOG)
        # for sample_idx, sample in enumerate(sample_list):
        #     print(sample_idx)
        #     func(sample, data_dict[str(sample["sample_idx"])], in_inference, sample_idx % self.args.num_gpus)
        return total_samples, complete_samples, correct_samples

    def flush_all_temp_files(self, data_sample_file, data_sample_dir, data_dict):
        json.dump(data_dict, open(data_sample_file, "w", encoding="utf-8"), ensure_ascii=False)
        existing_sample_filename_list = os.listdir(data_sample_dir)
        for sample_filename in existing_sample_filename_list:
            match = re.search(DATA_SAMPLE_TEMP_FILE_PATTERN, sample_filename)
            if match:
                sample_idx = str(int(match.group(1)))
                assert sample_idx in data_dict, "Error: temp file {:s} was not wrapped to the global file.".format(sample_filename)
            os.remove(os.path.join(data_sample_dir, sample_filename))
        os.rmdir(data_sample_dir)

    def generate_data(self, data_stage):
        data_sample_file, data_sample_dir, method_name, max_workers, in_inference = self.get_data_config(data_stage)
        print_with_indent("Starting generating {:s}...".format(str(data_stage)), indent=INDENT_INFO)
        start_time = time.time()
        subtitles = self.subtitle_data["subtitles"]
        if os.path.exists(data_sample_file):
            data_dict = json.load(open(data_sample_file, "r", encoding="utf-8"))
        else:
            data_dict = {str(subtitle_idx): None for subtitle_idx in range(subtitles)}
        os.makedirs(data_sample_dir, exist_ok=True)
        parameters = DataCuration.Parameters
        data_sampler = DataSampler({
            DataCuration.SampleType.WRONG: parameters.SAMPLING_RATIO_WRONG,
            DataCuration.SampleType.LOW_AND_CLOSE: parameters.SAMPLING_RATIO_LOW + parameters.SAMPLING_RATIO_CLOSE,
            DataCuration.SampleType.WRONG.LOW_ONLY: parameters.SAMPLING_RATIO_LOW,
            DataCuration.SampleType.WRONG.CLOSE_ONLY: parameters.SAMPLING_RATIO_CLOSE,
            DataCuration.SampleType.WRONG.CONFIDENT: parameters.SAMPLING_RATIO_CONFIDENT,
        }, sample_all=in_inference)
        existing_sample_filename_list = os.listdir(data_sample_dir)
        for subtitle_idx in range(subtitles):
            subtitle_object = self.subtitle_data["subtitle_list"][subtitle_idx]
            if subtitle_object["is_invalid"]:
                continue
            character_gt = subtitle_object["character_gt"]
            character_pred, sim_top1, sim_advantage = self.label_set.get_prediction_info(subtitle_idx)
            if self.drama_data.is_prediction_correct(character_gt, character_pred):
                if sim_top1 <= parameters.SIM_THRES_LOW and sim_advantage <= parameters.SIM_THRES_CLOSE:
                    sample_type = DataCuration.SampleType.LOW_AND_CLOSE
                elif sim_top1 <= parameters.SIM_THRES_LOW:
                    sample_type = DataCuration.SampleType.LOW_ONLY
                elif sim_advantage <= parameters.SIM_THRES_CLOSE:
                    sample_type = DataCuration.SampleType.CLOSE_ONLY
                else:
                    sample_type = DataCuration.SampleType.CONFIDENT
            else:
                sample_type = DataCuration.SampleType.WRONG
            sample_file = os.path.join(data_sample_dir, DATA_SAMPLE_TEMP_FILE(subtitle_idx))
            sample_exists = (data_dict[str(subtitle_idx)] is not None) or (sample_file in existing_sample_filename_list)
            data_sampler.add_new_sample_idx(sample_type, subtitle_idx, sample_file, sample_exists)
        sample_list = data_sampler.get_sample_list()
        print_with_indent("{:d} samples are to be collected.".format(len(sample_list)), indent=INDENT_LOG)
        total_samples, complete_samples, correct_samples = self.collect_all_data(
            data_dict, sample_list, method_name, max_workers, parameters, in_inference)
        self.flush_all_temp_files(data_sample_file, data_sample_dir, data_dict)
        print_with_indent("Finished collecting {:d} samples (complete/correct: {:d}/{:d}) of {:s}; {:0.6f} seconds elapsed.".format(
            total_samples, complete_samples, correct_samples, str(data_stage), time.time() - start_time), indent=INDENT_INFO)

    def curate_reasoning_data(self):
        print_module_title("Starting data curation...", is_start=True)
        self.load_pseudo_labels()
        if self.args.drama_flag_sft:
            self.generate_data(DataCuration.DataStage.SFT)
        if self.args.drama_flag_rl:
            self.generate_data(DataCuration.DataStage.RL)
        print_module_title("Finished data curation.", is_start=False)
        return self

    def perform_model_inference(self):
        print_module_title("Starting model inference...", is_start=True)
        self.load_pseudo_labels()
        if self.args.drama_flag_infer:
            self.generate_data(DataCuration.DataStage.INFER)
        print_module_title("Finished model inference.", is_start=False)
        return self

    def read_validation_file(result_file):
        result_file = os.path.join(target_dir, filename)
        data_list, total, corrects = [], 0, 0
        with open(result_file, "r") as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    data_list.append(json_object)
                    total += 1
                    if "score" in json_object:
                        corrects += json_object["score"]
                except json.JSONDecodeError as e:
                    print("Error decoding JSON from line: {:s}".format(line))
        return total, corrects

    def get_statistics(self, model_dir_list):
        self.load_pseudo_labels()
        subtitles = self.subtitle_data["subtitles"]
        character_gt, character_lp, character_lrm = [None] * subtitles, [None] * subtitles, [None] * subtitles
        duration, candidate_list = [0] * subtitles, [None] * subtitles
        total_samples, correct_lp, advantage_lp = 0, np.zeros((subtitles), dtype=np.int32), np.zeros((subtitles), dtype=np.float32)
        for subtitle_idx in range(subtitles):
            subtitle_object = self.subtitle_data["subtitle_list"][subtitle_idx]
            if subtitle_object["is_invalid"]:
                continue
            total_samples += 1
            character_gt[subtitle_idx] = subtitle_object["character_gt"]
            character_lp[subtitle_idx], _, advantage_lp[subtitle_idx] = self.label_set.get_prediction_info(subtitle_idx)
            candidate_list[subtitle_idx] = self.label_set.utterance_label[subtitle_idx]["rel_info"]["_list"]
            correct_lp[subtitle_idx] = self.drama_data.is_prediction_correct(
                character_gt[subtitle_idx], character_lp[subtitle_idx], candidate_list=candidate_list[subtitle_idx])
            duration[subtitle_idx] = subtitle_object["end_timestamp"] - subtitle_object["start_timestamp"]
        for model_dir in model_dir_list:
            print_with_indent("Evaluating model {:s}".format(os.path.basename(model_dir)), indent=INDENT_INFO)
            rl_starter_template_file = os.path.join(ASSET_PATH, RL_STARTER_FILE)
            rl_starter_replaced_file = os.path.join(model_dir, RL_STARTER_FILE)
            set_args_from_config_file(rl_starter_template_file, rl_starter_replaced_file, self.args, overwrite=True, allow_no_key_diff=True)
            print(self.args)
            print("rl_group_size\t{:d}".format(int(self.args.rl_group_size)))
            print("total_epochs\t{:d}".format(int(self.args.rl_num_epochs)))
            print("batch_size\t{:d}".format(int(self.args.rl_batch_size)))
            print("temperature\t{:0.4f}".format(float(self.args.rl_temperature)))
            print("top_p\t{:0.4f}".format(float(self.args.rl_top_p)))
            print("presence_penalty\t{:0.4f}".format(float(self.args.rl_presence_penalty)))
            print("frequency_penalty\t{:0.4f}".format(float(self.args.rl_frequency_penalty)))
            print("kl_loss_coef\t{:0.4f}".format(float(self.args.rl_kl_loss_coef)))
            print("kl_coef\t{:0.4f}".format(float(self.args.rl_kl_coef)))
            print("lr\t{:0.4f}".format(float(self.args.rl_learning_rate)))
            print("penalty_length\t{:0.4f}".format(float(self.args.rl_penalty_length)))
            print("penalty_duplicate\t{:0.4f}".format(float(self.args.rl_penalty_duplicate)))
            # if not os.path.basename(model_dir).startswith("model_rl_20260327"):
            #     continue
            rollout_dir = os.path.join(model_dir, "rollout_data", "val")
            result_dir = os.path.join(model_dir, "test_results_{:0.2f}_{:0.2f}".format(self.args.infer_temperature, self.args.infer_top_p))
            if not os.path.exists(result_dir):
                result_dir = os.path.join(model_dir, "test_results")
                if not os.path.exists(result_dir):
                    continue
            snapshot_idx_list = []
            for snapshot_name in os.listdir(result_dir):
                match = re.search(RL_SNAPSHOT_DIR_PATTERN, snapshot_name)
                if not match:
                    continue
                snapshot_idx_list.append(int(match.group(1)))
            for snapshot_idx in sorted(snapshot_idx_list):
                rollout_file = os.path.join(rollout_dir, "{:d}.jsonl".format(snapshot_idx))
                total_val, corrects_val = 0, 0
                with open(rollout_file, "r") as file:
                    for line in file:
                        try:
                            json_object = json.loads(line)
                            total_val += 1
                            if "score" in json_object:
                                corrects_val += json_object["score"]
                        except json.JSONDecodeError as e:
                            print("Error decoding JSON from line: {:s}".format(line))
                snapshot_name = RL_SNAPSHOT_DIR(snapshot_idx)
                infer_data_filename = INFER_DATA_FILE(self.args.drama_name, self.args.infer_data_str)
                infer_data_file = os.path.join(result_dir, snapshot_name, infer_data_filename)
                if not os.path.exists(infer_data_file):
                    continue
                # print_with_indent("Evaluating snapshot {:s}".format(infer_data_file), indent=INDENT_LOG)
                infer_data = json.load(open(infer_data_file, "r", encoding="utf-8"))
                correct_lrm = np.zeros((subtitles), dtype=np.int32)
                incompletes = 0
                # helps, hurts = 0, 0
                for subtitle_idx in range(subtitles):
                    if character_gt[subtitle_idx] is None:
                        continue
                    character_lrm[subtitle_idx] = character_lp[subtitle_idx]
                    infer_sample = infer_data[str(subtitle_idx)]
                    assert infer_sample is not None and "answer" in infer_sample, "E!!!"
                    if infer_sample["answer"] is not None:
                        character_lrm[subtitle_idx] = infer_sample["answer"]
                    else:
                        incompletes += 1
                    correct_lrm[subtitle_idx] = self.drama_data.is_prediction_correct(
                        character_gt[subtitle_idx], character_lrm[subtitle_idx], candidate_list=candidate_list[subtitle_idx])
                #     if advantage_lp[subtitle_idx] >= 0.04:
                #         continue
                #     if not 1000 <= duration[subtitle_idx] < 2000:
                #         continue
                #     if correct_lrm[subtitle_idx] == 1 and correct_lp[subtitle_idx] == 0:
                #         helps += 1
                #     if correct_lp[subtitle_idx] == 1 and correct_lrm[subtitle_idx] == 0:
                #         hurts += 1
                # print(helps, hurts)
                # continue
                THRESHOLD_COUNT = 12
                threshold, corrects, accuracy = [0] * THRESHOLD_COUNT, [0] * THRESHOLD_COUNT, [0] * THRESHOLD_COUNT
                for threshold_idx in range(THRESHOLD_COUNT):
                    threshold[threshold_idx] = threshold_idx * 0.02 if threshold_idx < THRESHOLD_COUNT - 1 else 1.0
                    corrects[threshold_idx] = np.sum(correct_lp[advantage_lp >= threshold[threshold_idx]]) + \
                        np.sum(correct_lrm[advantage_lp < threshold[threshold_idx]])
                    accuracy[threshold_idx] = float(corrects[threshold_idx]) / total_samples
                # corrects = [corrects_val] + corrects
                # accuracy = [float(corrects_val) / total_val] + accuracy
                # print(pd.DataFrame([accuracy], columns=["val"] + threshold).to_string(index=False, header=False))
                print("{:d}\t{:0.2f}".format(snapshot_idx, float(corrects_val) / total_val * 100), end="")
                for threshold_idx in range(THRESHOLD_COUNT):
                    print("\t{:0.2f}".format(accuracy[threshold_idx] * 100), end="")
                print("\t{:d}".format(incompletes))
                # print()
            print()

    def error_analysis(self, model_dir_list):
        self.load_pseudo_labels()
        subtitles = self.subtitle_data["subtitles"]
        character_gt, character_lp, character_lrm = [None] * subtitles, [None] * subtitles, [None] * subtitles
        duration, candidate_list = [0] * subtitles, [None] * subtitles
        total_samples, correct_lp, advantage_lp = 0, np.zeros((subtitles), dtype=np.int32), np.zeros((subtitles), dtype=np.float32)
        for subtitle_idx in range(subtitles):
            subtitle_object = self.subtitle_data["subtitle_list"][subtitle_idx]
            if subtitle_object["is_invalid"]:
                continue
            total_samples += 1
            character_gt[subtitle_idx] = subtitle_object["character_gt"]
            character_lp[subtitle_idx], _, advantage_lp[subtitle_idx] = self.label_set.get_prediction_info(subtitle_idx)
            candidate_list[subtitle_idx] = self.label_set.utterance_label[subtitle_idx]["rel_info"]["_list"]
            correct_lp[subtitle_idx] = self.drama_data.is_prediction_correct(
                character_gt[subtitle_idx], character_lp[subtitle_idx], candidate_list=candidate_list[subtitle_idx])
            duration[subtitle_idx] = subtitle_object["end_timestamp"] - subtitle_object["start_timestamp"]
        for model_dir in model_dir_list:
            print_with_indent("Evaluating model {:s}".format(os.path.basename(model_dir)), indent=INDENT_INFO)
            rl_starter_template_file = os.path.join(ASSET_PATH, RL_STARTER_FILE)
            rl_starter_replaced_file = os.path.join(model_dir, RL_STARTER_FILE)
            set_args_from_config_file(rl_starter_template_file, rl_starter_replaced_file, self.args, overwrite=True, allow_no_key_diff=True)
            print(self.args)
            print("rl_group_size\t{:d}".format(int(self.args.rl_group_size)))
            print("total_epochs\t{:d}".format(int(self.args.rl_num_epochs)))
            print("batch_size\t{:d}".format(int(self.args.rl_batch_size)))
            print("temperature\t{:0.4f}".format(float(self.args.rl_temperature)))
            print("top_p\t{:0.4f}".format(float(self.args.rl_top_p)))
            print("presence_penalty\t{:0.4f}".format(float(self.args.rl_presence_penalty)))
            print("frequency_penalty\t{:0.4f}".format(float(self.args.rl_frequency_penalty)))
            print("kl_loss_coef\t{:0.4f}".format(float(self.args.rl_kl_loss_coef)))
            print("kl_coef\t{:0.4f}".format(float(self.args.rl_kl_coef)))
            print("lr\t{:0.4f}".format(float(self.args.rl_learning_rate)))
            print("penalty_length\t{:0.4f}".format(float(self.args.rl_penalty_length)))
            print("penalty_duplicate\t{:0.4f}".format(float(self.args.rl_penalty_duplicate)))
            result_dir = os.path.join(model_dir, "test_results_{:0.2f}_{:0.2f}".format(self.args.infer_temperature, self.args.infer_top_p))
            if not os.path.exists(result_dir):
                result_dir = os.path.join(model_dir, "test_results")
                if not os.path.exists(result_dir):
                    continue
            snapshot_idx_list = []
            for snapshot_name in os.listdir(result_dir):
                match = re.search(RL_SNAPSHOT_DIR_PATTERN, snapshot_name)
                if not match:
                    continue
                snapshot_idx_list.append(int(match.group(1)))
            for snapshot_idx in [sorted(snapshot_idx_list)[-1]]:
                snapshot_name = RL_SNAPSHOT_DIR(snapshot_idx)
                infer_data_filename = INFER_DATA_FILE(self.args.drama_name, self.args.infer_data_str)
                infer_data_file = os.path.join(result_dir, snapshot_name, infer_data_filename)
                if not os.path.exists(infer_data_file):
                    continue
                # print_with_indent("Evaluating snapshot {:s}".format(infer_data_file), indent=INDENT_LOG)
                infer_data = json.load(open(infer_data_file, "r", encoding="utf-8"))
                correct_lrm = np.zeros((subtitles), dtype=np.int32)
                incompletes = 0
                for subtitle_idx in range(subtitles):
                    if character_gt[subtitle_idx] is None:
                        continue
                    infer_sample = infer_data[str(subtitle_idx)]
                    assert infer_sample is not None and "answer" in infer_sample, "E!!!"
                    if infer_sample["answer"] is not None:
                        character_lrm[subtitle_idx] = infer_sample["answer"]
                        correct_lrm[subtitle_idx] = self.drama_data.is_prediction_correct(
                            character_gt[subtitle_idx], character_lrm[subtitle_idx], candidate_list=candidate_list[subtitle_idx])
                    else:
                        character_lrm[subtitle_idx] = character_lp[subtitle_idx]
                        correct_lrm[subtitle_idx] = False
                        incompletes += 1
                    if correct_lp[subtitle_idx] and not correct_lrm[subtitle_idx]:
                        print("=" * 50)
                        if advantage_lp[subtitle_idx] >= 0.10:
                            print("Error type = CONFIDENT, LP advantage = {:0.4f}".format(advantage_lp[subtitle_idx]))
                        else:
                            print("Error type = CLOSE, LP advantage = {:0.4f}".format(advantage_lp[subtitle_idx]))
                        print("Subtitle idx = {:d}".format(subtitle_idx))
                        print("-" * 50)
                        print(self.subtitle_data["subtitle_list"][subtitle_idx])
                        print("-" * 50)
                        print(infer_sample)
                        print("-" * 50)
            print()
