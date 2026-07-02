import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd
import json
import re
import random
from enum import Enum

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from large_reasoning_model.toolset import Toolset


class ModelTraining:

    class TargetModel(Enum):
        Q3_8B = "q3_8b"
        def __str__(self):
            return self.value

    SFT_KEY_PARAMETER_LIST = [
        "sft_init_model_dir",
        "sft_deepspeed_config_file",
        "sft_dataset_dir",
        "sft_dataset_name",
        "sft_validation_ratio",
        "sft_learning_rate",
        "sft_num_epochs",
    ]

    RL_KEY_PARAMETER_LIST = [
        "sft_model_dir",
        "training_data_file",
        "validation_data_file",
        "rl_validation_ratio",
        "rl_learning_rate",
        "rl_num_epochs",
        "rl_group_size",
        "rl_batch_size",
        "rl_temperature",
        "rl_top_p",
        "rl_presence_penalty",
        "rl_frequency_penalty",
        "rl_max_turns",
        "rl_kl_loss_coef",
        "rl_kl_coef",
        "rl_penalty_length",
        "rl_penalty_duplicate",
        "toolset_name",
    ]

    def __init__(self, project, args, args_list_sft, args_list_rl):
        self.project = project
        self.args = args
        self.args_list_sft = args_list_sft
        self.args_list_rl = args_list_rl

    @staticmethod
    def find_compatible_output_dir(template_file, target_dir, prefix, config_filename, key_arg_dict):
        for model_name in sorted(os.listdir(target_dir), reverse=True):
            model_dir = os.path.join(target_dir, model_name)
            if not os.path.isdir(model_dir) or not model_name.startswith(prefix):
                continue
            replaced_file = os.path.join(model_dir, config_filename)
            if not os.path.exists(replaced_file):
                continue
            keyword_dict = match_template_file(template_file, replaced_file)
            if keyword_dict is None:
                continue
            matched = True
            for key, value in key_arg_dict.items():
                if key not in keyword_dict:
                    matched = False
                try:
                    value1 = float(value)
                    value2 = float(keyword_dict[key])
                    matched = (value1 == value2)
                except ValueError:
                    matched = (value == keyword_dict[key])
                if not matched:
                    break
            if matched:
                return os.path.join(target_dir, model_name)
        return None

    def get_model_output_dir(self, template_file, cached_models_dir, output_dir, prefix, config_file, key_parameter_list, new_model_name,
                             ignore_cached_models=False, ignore_training_models=False):
        key_arg_dict = {arg_name: getattr(self.args, arg_name) for arg_name in key_parameter_list}
        model_output_dir = ModelTraining.find_compatible_output_dir(
            template_file, cached_models_dir, prefix, config_file, key_arg_dict)
        if not ignore_cached_models and model_output_dir is not None:
            return model_output_dir, True, False
        model_output_dir = ModelTraining.find_compatible_output_dir(
            template_file, output_dir, prefix, config_file, key_arg_dict)
        if not ignore_training_models and model_output_dir is not None:
            return model_output_dir, False, False
        model_output_dir = os.path.join(output_dir, new_model_name)
        return model_output_dir, False, True

    @staticmethod
    def copy_prompt_for_backup(src_prompt_file, dst_model_dir):
        dst_prompt_file = os.path.join(dst_model_dir, os.path.basename(src_prompt_file))
        copy_command = "cp {:s} {:s}".format(src_prompt_file, dst_prompt_file)
        subprocess.run(copy_command, shell=True)

    def prepare_sft_data(self):
        # Step 1: Wrap up data into a single file.
        sft_data_all_file = os.path.join(self.args.version_dir, SFT_DATA_ALL_FILE(self.args.sft_data_str))
        if not os.path.exists(sft_data_all_file):
            print_with_indent("Wrapping SFT data into one file...", indent=INDENT_INFO)
            start_time = time.time()
            sft_data_all_list, used_tool_dict = [], {}
            for args in self.args_list_sft:
                sft_data_file = os.path.join(self.args.drama_data_dir, SFT_DATA_FILE(args.drama_name, args.sft_data_str))
                sft_data_list = json.load(open(sft_data_file, "r", encoding="utf-8"))
                for sample_idx in sft_data_list:
                    sample_data = sft_data_list[sample_idx]
                    if sample_data is None:
                        continue
                    messages = len(sample_data["message_list"])
                    assert len(sample_data["message_list"]) >= 3, "Error: an SFT trajectory must have at least 3 messages."
                    assert len(sample_data["message_list"]) % 2 == 1, "Error: an SFT trajectory must have an odd number of messages."
                    assert sample_data["message_list"][0]["role"] == "system", "Error: the first message must be system message."
                    system_message, history_message_list = sample_data["message_list"][0]["content"], []
                    input_message, output_message = None, None
                    for message_idx in range(1, len(sample_data["message_list"])):
                        if message_idx % 2 == 1:
                            assert sample_data["message_list"][message_idx]["role"] == "user", "Error: the odd-numbered messages must be user messages."
                            if sample_data["useful_message_mask"][message_idx]:
                                input_message = sample_data["message_list"][message_idx]["content"]
                        if message_idx % 2 == 0:
                            assert sample_data["message_list"][message_idx]["role"] == "assistant", "Error: the even-numbered messages must be assistant messages."
                            if sample_data["useful_message_mask"][message_idx]:
                                output_message = sample_data["message_list"][message_idx]["content"]
                            else:
                                continue
                            assert input_message is not None, "Error: input message was not found."
                            sft_sample = {
                                "instruction": input_message,
                                "system": system_message,
                                "input": "",
                                "output": output_message,
                                "history": history_message_list.copy(),
                            }
                            sft_data_all_list.append(sft_sample)
                            decoded_content = Toolset.decode_content(output_message)[2]
                            if "tool" in decoded_content:
                                tool_name = decoded_content["tool"]
                                if not tool_name in used_tool_dict:
                                    used_tool_dict.update({tool_name: 1})
                                else:
                                    used_tool_dict[tool_name] += 1
                            history_message_list.append([input_message, output_message])
                            input_message, output_message = None, None
            json.dump(sft_data_all_list, open(sft_data_all_file, "w", encoding="utf-8"), ensure_ascii=False)
            print_with_indent("Finished wrapping SFT data with {:d} training samples; used tools: {:s}; {:0.6f} seconds elapsed.".format(
                len(sft_data_all_list), str(used_tool_dict), time.time() - start_time), indent=INDENT_INFO)
        # Step 2: Write the sft_data_info.json file.
        dataset_name = "sft_training_data"
        data_file_name = SFT_DATA_ALL_FILE(self.args.sft_data_str)
        sft_data_info = {dataset_name: {
            "file_name": data_file_name,
            "columns": {
                "history": "history",
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            }
        }}
        sft_data_info_file = os.path.join(self.args.version_dir, SFT_DATA_INFO_FILE)
        json.dump(sft_data_info, open(sft_data_info_file, "w", encoding="utf-8"), ensure_ascii=False)
        # Step 3: Write the sft_config.yaml file.
        root_path = self.args.root_absolute_path
        cached_models_dir = os.path.join(root_path, CACHED_MODELS_PATH)
        sft_config_template_file = os.path.join(root_path, ASSET_PATH, SFT_CONFIG_FILE)
        self.args.sft_init_model_dir = os.path.join(root_path, ASSET_PATH, MODEL_NAME_DICT_LOCAL[self.args.target_model.value])
        self.args.sft_deepspeed_config_file = os.path.join(root_path, DEEPSPEED_CONFIG_FILE)
        self.args.sft_dataset_dir = os.path.join(root_path, self.args.version_dir)
        self.args.sft_dataset_name = dataset_name
        self.args.sft_output_dir, self.args.sft_training_done, self.args.sft_training_anew = self.get_model_output_dir(
            sft_config_template_file, cached_models_dir, OUTPUT_PATH(self.args.version_name), SFT_MODEL_PREFIX,
            SFT_CONFIG_FILE, ModelTraining.SFT_KEY_PARAMETER_LIST, SFT_MODEL_BASEDIR(get_current_datetime()))
        sft_config_replaced_file = os.path.join(self.args.sft_output_dir, SFT_CONFIG_FILE)
        if self.args.sft_training_anew:
            os.makedirs(self.args.sft_output_dir, exist_ok=True)
            replace_template_file(sft_config_template_file, sft_config_replaced_file, self.args)
        else:
            set_args_from_config_file(sft_config_template_file, sft_config_replaced_file, self.args)

    ## not tested!!!
    def run_sft_training(self):
        llama_factory_dir = os.path.join(self.args.root_absolute_path, LLAMA_FACTORY_DIR)
        sft_config_file = os.path.join(self.args.sft_output_dir, SFT_CONFIG_FILE)
        file_parts = 0
        while True:
            file_parts += 1
            sft_log_file = os.path.join(self.args.sft_output_dir, SFT_LOG_FILE(file_parts))
            if not os.path.exists(sft_log_file):
                break
        shell_command = "llamafactory-cli train {:s} 2>&1 | tee {:s}".format(sft_config_file, sft_log_file)
        subprocess.run(shell_command, cwd=llama_factory_dir, shell=True)

    ## not tested!!!
    def backup_sft_training_results(self):
        ...

    def perform_sft_training(self):
        print_module_title("Starting SFT training...", is_start=True)
        self.prepare_sft_data()
        if not self.args.sft_training_done:
            if self.args.sft_training_anew:
                print_with_indent("Starting new training at {:s}".format(self.args.sft_output_dir), indent=INDENT_INFO)
            else:
                print_with_indent("Resuming training at {:s}".format(self.args.sft_output_dir), indent=INDENT_INFO)
            ModelTraining.copy_code_for_backup(self.args.root_absolute_path, self.args.sft_output_dir)
            self.run_sft_training()
            self.backup_sft_training_results()
        else:
            print_with_indent("Found complete training at {:s}".format(self.args.sft_output_dir), indent=INDENT_INFO)
        print_module_title("Finished SFT training.", is_start=False)

    def prepare_rl_data(self):
        # Step 1: Wrap up data into a single file.
        rl_training_data_file = os.path.join(self.args.version_dir, RL_DATA_ALL_FILE(self.args.rl_training_data_str))
        rl_validation_data_file = os.path.join(self.args.version_dir, RL_DATA_ALL_FILE(self.args.rl_validation_data_str))
        data_source_name = DEFAULT_DATASOURCE_NAME
        if not os.path.exists(rl_training_data_file) or not os.path.exists(rl_validation_data_file):
            print_with_indent("Wrapping RL data into one file...", indent=INDENT_INFO)
            start_time = time.time()
            rl_data_all_list = []
            for args in self.args_list_rl:
                rl_data_file = os.path.join(self.args.drama_data_dir, RL_DATA_FILE(args.drama_name, args.rl_data_str))
                rl_data_list = json.load(open(rl_data_file, "r", encoding="utf-8"))
                for sample_idx in rl_data_list:
                    sample_data = rl_data_list[sample_idx]
                    if sample_data is None:
                        continue
                    message_list = sample_data["message_list"]
                    assert len(message_list) >= 2, "Error: an RL data must have at least 2 messages."
                    assert message_list[0]["role"] == "system", "Error: the first message must be system message."
                    assert message_list[1]["role"] == "user", "Error: the second message must be user message."
                    system_message, user_message = message_list[0]["content"], message_list[1]["content"]
                    ground_truth = sample_data["answer"] if sample_data["answer"] in sample_data["candidate_list"] else CHARACTER_NAME_OTHERS[args.language]
                    rl_sample = {
                        "data_source": data_source_name,
                        "prompt": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                        "ability": "commonsense", ## check
                        "reward_model": {
                            "style": "rule", ## check
                            "ground_truth": {
                                "character_gt": ground_truth,
                                "candidate_list": sample_data["candidate_list"],
                                "name_others": CHARACTER_NAME_OTHERS[args.language],
                            }
                        },
                        "extra_info": {
                            "drama_name": args.drama_name,
                            "subtitle_idx": sample_data["subtitle_idx"],
                            "sample_type": sample_data["sample_type"],
                        }
                    }
                    rl_data_all_list.append(rl_sample)
            idx_list_all = list(range(len(rl_data_all_list)))
            idx_list_validation = sorted(random.sample(idx_list_all, max(1, int(len(rl_data_all_list) * self.args.rl_validation_ratio + 0.5))))
            idx_list_training = [idx for idx in idx_list_all if not idx in idx_list_validation]
            rl_data_training = pd.DataFrame([sample for idx, sample in enumerate(rl_data_all_list) if idx in idx_list_training])
            rl_data_validation = pd.DataFrame([sample for idx, sample in enumerate(rl_data_all_list) if idx in idx_list_validation])
            rl_data_training.to_parquet(rl_training_data_file, index=False)
            rl_data_validation.to_parquet(rl_validation_data_file, index=False)
            print_with_indent("Finished wrapping RL data with {:d}/{:d} training/validation samples; {:0.6f} seconds elapsed.".format(
                len(idx_list_training), len(idx_list_validation), time.time() - start_time), indent=INDENT_INFO)
        # Step 2: Write the rl_starter.sh file.
        root_path = self.args.root_absolute_path
        cached_models_dir = os.path.join(root_path, CACHED_MODELS_PATH)
        rl_starter_template_file = os.path.join(ASSET_PATH, RL_STARTER_FILE)
        self.args.sft_model_dir = self.args.sft_output_dir
        self.args.training_data_file = os.path.join(root_path, rl_training_data_file)
        self.args.validation_data_file = os.path.join(root_path, rl_validation_data_file)
        self.args.toolset_name = DEFAULT_TOOLSET_NAME
        self.args.rl_output_dir, self.args.rl_training_done, self.args.rl_training_anew = self.get_model_output_dir(
            rl_starter_template_file, cached_models_dir, OUTPUT_PATH(self.args.version_name), RL_MODEL_PREFIX,
            RL_STARTER_FILE, ModelTraining.RL_KEY_PARAMETER_LIST, RL_MODEL_BASEDIR(get_current_datetime()),
            ignore_cached_models=True, ignore_training_models=True)
        rl_starter_replaced_file = os.path.join(self.args.rl_output_dir, RL_STARTER_FILE)
        self.args.rl_output_path = os.path.dirname(self.args.rl_output_dir)
        self.args.rl_model_name = os.path.basename(self.args.rl_output_dir)
        if self.args.rl_training_anew:
            os.makedirs(self.args.rl_output_dir, exist_ok=True)
            self.args.rollout_training_dir = ROLLOUT_TRAINING_DIR(self.args.rl_output_path, self.args.rl_model_name)
            self.args.rollout_validation_dir = ROLLOUT_VALIDATION_DIR(self.args.rl_output_path, self.args.rl_model_name)
            self.args.drama_data_dir = self.args.drama_data_dir
            self.args.version_dir = self.args.version_dir
            self.args.drama_name_list = ",".join([args.drama_name for args in self.args_list_rl])
            self.args.label_prop_option_list = ",".join([args.label_prop_str for args in self.args_list_rl])
            replace_template_file(rl_starter_template_file, rl_starter_replaced_file, self.args)
        else:
            ...
            set_args_from_config_file(rl_starter_template_file, rl_starter_replaced_file, self.args)

    def run_rl_training(self):
        verl_tool_dir = os.path.join(self.args.root_absolute_path, VERL_TOOL_DIR)
        rl_starter_file = os.path.join(self.args.rl_output_dir, RL_STARTER_FILE)
        file_parts = 0
        while True:
            file_parts += 1
            rl_log_file = os.path.join(self.args.rl_output_dir, RL_LOG_FILE(file_parts))
            if not os.path.exists(rl_log_file):
                break
        shell_command = "bash {:s} 2>&1 | tee {:s}".format(rl_starter_file, rl_log_file)
        subprocess.run(shell_command, cwd=verl_tool_dir, shell=True)

    def backup_rl_training_results(self, all_snapshots=False, all_status=False):
        snapshot_list = []
        for rollout_filename in os.listdir(self.args.rollout_validation_dir):
            match = re.search(r"(\d+)\.jsonl", rollout_filename)
            if not match:
                continue
            snapshot_idx = int(match.group(1))
            snapshot_dir = os.path.join(self.args.rl_output_dir, RL_SNAPSHOT_DIR(snapshot_idx))
            assert os.path.exists(snapshot_dir), "Error: snapshot {:s} does not exist.".format(snapshot_dir)
            rollout_file = os.path.join(self.args.rollout_validation_dir, rollout_filename)
            total_count, total_rewards = 0, 0.0
            with open(rollout_file, "r") as file:
                for line in file:
                    json_object = json.loads(line)
                    total_count += 1
                    if "reward" in json_object:
                        total_rewards += json_object["reward"]
            snapshot_list.append({
                "snapshot_idx": snapshot_idx,
                "snapshot_dir": snapshot_dir,
                "total_count": total_count,
                "total_rewards": total_rewards,
            })
        sorted_snapshot_list = sorted(snapshot_list, key=lambda item: item["total_rewards"], reverse=True)
        all_cached_models_dir = os.path.join(self.args.root_absolute_path, CACHED_MODELS_PATH)
        cached_model_dir = os.path.join(all_cached_models_dir, self.args.rl_model_name)
        os.makedirs(cached_model_dir, exist_ok=True)
        all_snapshots_dir = os.path.join(cached_model_dir, "snapshots")
        os.makedirs(all_snapshots_dir, exist_ok=True)
        # Part 1: copy trained models.
        snapshot_dir_list = []
        for sorted_idx, snapshot_item in enumerate(sorted_snapshot_list):
            snapshot_idx = snapshot_item["snapshot_idx"]
            snapshot_dir = RL_SNAPSHOT_DIR(snapshot_idx)
            snapshot_dir_list.append(snapshot_dir)
            if not all_snapshots and sorted_idx > 0:
                continue
            snapshot_dir_src = os.path.join(self.args.rl_output_dir, snapshot_dir, "actor/huggingface/")
            snapshot_dir_dst = os.path.join(all_snapshots_dir, snapshot_dir)
            rsync_command = "rsync -av --inplace --size-only --progress {:s} {:s}".format(
                snapshot_dir_src, snapshot_dir_dst)
            subprocess.run(rsync_command, shell=True)
        # Part 2: copy rollout data.
        rollout_dir_dst = os.path.join(cached_model_dir, "rollout_data/")
        rsync_command = "rsync -av --inplace --size-only --progress {:s} {:s}".format(
            self.args.rollout_training_dir, rollout_dir_dst)
        subprocess.run(rsync_command, shell=True)
        rsync_command = "rsync -av --inplace --size-only --progress {:s} {:s}".format(
            self.args.rollout_validation_dir, rollout_dir_dst)
        subprocess.run(rsync_command, shell=True)
        # Part 3: copy others.
        rsync_excluded_list = " ".join(["--exclude {:s}".format(excluded_name) for excluded_name in snapshot_dir_list])
        rsync_command = "rsync -av --inplace --size-only --progress {:s} {:s} {:s}".format(
            rsync_excluded_list, self.args.rl_output_dir, all_cached_models_dir)
        subprocess.run(rsync_command, shell=True)

    def perform_rl_training(self):
        print_module_title("Starting RL training...", is_start=True)
        self.prepare_rl_data()
        if not self.args.rl_training_done:
            if self.args.rl_training_anew:
                print_with_indent("Starting new training at {:s}".format(self.args.rl_output_dir), indent=INDENT_INFO)
            else:
                print_with_indent("Resuming training at {:s}".format(self.args.rl_output_dir), indent=INDENT_INFO)
            src_prompt_file = os.path.join(self.args.root_absolute_path, "large_reasoning_model/llm_prompts.py")
            ModelTraining.copy_prompt_for_backup(src_prompt_file, self.args.rl_output_dir)
            self.run_rl_training()
            self.backup_rl_training_results(all_snapshots=True)
        else:
            print_with_indent("Found complete training at {:s}".format(self.args.rl_output_dir), indent=INDENT_INFO)
        print_module_title("Finished RL training...", is_start=True)

    def train_reasoning_model(self):
        self.perform_sft_training()
        self.perform_rl_training()
