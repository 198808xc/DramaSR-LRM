import os
import sys
import json
import re
import argparse
from enum import Enum

from utils import *
from data_loader import DramaData
from label_propagation.label_propagation import LabelPropagation
from large_reasoning_model.data_curation import DataCuration
from large_reasoning_model.model_training import ModelTraining
from large_reasoning_model.model_inference import ModelInference


class SubArguments:

    def __init__(self):
        pass

    def __repr__(self):
        return str({attr: getattr(self, attr) for attr in dir(self) if not callable(
            getattr(self, attr)) and not attr.startswith("__")})

    def get_sub_args(self, args, drama_idx: int):
        # Paths
        setattr(self, "drama_data_dir", args.drama_data_dir)
        setattr(self, "drama_dir", args.drama_dir[drama_idx])
        setattr(self, "version_dir", args.version_dir)
        # Drama info
        setattr(self, "version_name", args.version_name)
        setattr(self, "drama_name", args.drama_list[drama_idx])
        setattr(self, "language", args.language[drama_idx])
        setattr(self, "episodes", args.episodes[drama_idx])
        setattr(self, "drama_flag_sft", args.drama_flag_sft[drama_idx])
        setattr(self, "drama_flag_rl", args.drama_flag_rl[drama_idx])
        setattr(self, "drama_flag_infer", args.drama_flag_infer[drama_idx])
        # Data arguments
        setattr(self, "init_seeds_option", args.init_seeds_option)
        setattr(self, "init_seeds_str", args.init_seeds_option)
        setattr(self, "label_prop_option", args.label_prop_option)
        setattr(self, "label_prop_str", "{:s}_{:s}".format(
            self.init_seeds_option, self.label_prop_option))
        setattr(self, "sft_data_option", args.sft_data_option)
        setattr(self, "sft_data_str", "{:s}_{:s}_{:s}".format(
            self.init_seeds_option, self.label_prop_option, self.sft_data_option))
        setattr(self, "sft_workers", args.sft_workers)
        setattr(self, "rl_data_option", args.rl_data_option)
        setattr(self, "rl_data_str", "{:s}_{:s}_{:s}".format(
            self.init_seeds_option, self.label_prop_option, self.rl_data_option))
        setattr(self, "rl_workers", args.rl_workers)
        setattr(self, "api_key", args.api_key)
        setattr(self, "base_url", args.base_url)
        setattr(self, "enforce_refresh", args.enforce_refresh)
        setattr(self, "enforce_retrial", args.enforce_retrial)
        # Training arguments
        setattr(self, "num_gpus", args.num_gpus)
        setattr(self, "target_model", args.target_model)
        setattr(self, "sft_validation_ratio", args.sft_validation_ratio)
        setattr(self, "sft_enforce_retraining", args.sft_enforce_retraining)
        setattr(self, "sft_learning_rate", args.sft_learning_rate)
        setattr(self, "sft_num_epochs", args.sft_num_epochs)
        setattr(self, "rl_validation_ratio", args.rl_validation_ratio)
        setattr(self, "rl_enforce_retraining", args.rl_enforce_retraining)
        setattr(self, "rl_learning_rate", args.rl_learning_rate)
        setattr(self, "rl_num_epochs", args.rl_num_epochs)
        setattr(self, "rl_group_size", args.rl_group_size)
        setattr(self, "rl_batch_size", args.rl_batch_size)
        setattr(self, "rl_temperature", args.rl_temperature)
        setattr(self, "rl_top_p", args.rl_top_p)
        setattr(self, "rl_presence_penalty", args.rl_presence_penalty)
        setattr(self, "rl_frequency_penalty", args.rl_frequency_penalty)
        setattr(self, "rl_max_turns", args.rl_max_turns)
        setattr(self, "rl_kl_loss_coef", args.rl_kl_loss_coef)
        setattr(self, "rl_kl_coef", args.rl_kl_coef)
        setattr(self, "rl_penalty_length", args.rl_penalty_length)
        setattr(self, "rl_penalty_duplicate", args.rl_penalty_duplicate)
        # Inference arguments
        setattr(self, "infer_data_str", "{:s}_{:s}_{:s}_{:s}_{:s}".format(
            self.init_seeds_option, self.label_prop_option, self.sft_data_option, self.rl_data_option, self.target_model))
        setattr(self, "infer_workers", args.infer_workers)
        setattr(self, "api_key_local", args.api_key_local)
        setattr(self, "base_port_id", args.base_port_id)
        setattr(self, "infer_temperature", args.infer_temperature)
        setattr(self, "infer_top_p", args.infer_top_p)
        return self


class ProjectDramaSR:

    VERSION_DIR_PATTERN = "(.*?)\_[0-9]{8}\_[0-9]{6}"
    VERSION_DIR_TIMESTAMP_LENGTH = 15
    VERSION_META_FILE = "meta.json"
    VERSION_META_ARGS = ["drama_list_sft", "drama_list_rl",
                         "init_seeds_option", "label_prop_option", "sft_data_option", "rl_data_option"]

    PROJECT_PROGRESS_INFO = [
    ]

    class WorkingStage(Enum):
        DATA_PREPARATION = "data_preparation"
        DATA_CURATION = "data_curation"
        MODEL_TRAINING = "model_training"
        MODEL_INFERENCE = "model_inference"
        RESULT_STATISTICS = "result_statistics"

    class ExistingDataType(Enum):
        NONE = "none"
        LAST = "last"
        COMPLETE = "complete"

    def __init__(self, args):
        self.args = args
        os.makedirs(self.args.workspace_dir, exist_ok=True)
        self.args.version_dir, self.args.version_name = None, None
        self.drama_data = None
        self.label_propagator = None
        self.data_curator = None
        self.model_trainer = None
        self.model_inferer = None

    def get_new_version_dir(self) -> str:
        return "{:s}_{:s}".format(self.args.version_identifier, get_current_datetime())

    def create_new_version_dir(self):
        new_version_name = self.get_new_version_dir()
        new_version_dir = os.path.join(self.args.workspace_dir, new_version_name)
        os.makedirs(new_version_dir, exist_ok=True)
        meta = {arg_name: str(getattr(self.args, arg_name)) for arg_name in ProjectDramaSR.VERSION_META_ARGS}
        version_meta_file = os.path.join(new_version_dir, ProjectDramaSR.VERSION_META_FILE)
        json.dump(meta, open(version_meta_file, "w", encoding="utf-8"), ensure_ascii=False)
        return new_version_dir, new_version_name

    def meta_info_matched(self, version_dir: str) -> int:
        version_meta_file = os.path.join(version_dir, ProjectDramaSR.VERSION_META_FILE)
        if not os.path.exists(version_meta_file):
            return False
        version_meta = json.load(open(version_meta_file, "r", encoding="utf-8"))
        for arg_name in ProjectDramaSR.VERSION_META_ARGS:
            if arg_name not in version_meta:
                return False
            new_arg, version_arg = str(getattr(self.args, arg_name)), str(version_meta[arg_name])
            # print(new_arg, version_arg, new_arg != version_arg)
            if new_arg != version_arg:
                return False
        return True

    def get_project_progress(self, version_dir: str) -> int:
        progress = 0
        for drama_name in self.args.drama_list:
            for (progress_file, option_name) in ProjectDramaSR.PROJECT_PROGRESS_INFO:
                if not os.path.exists(os.path.join(version_dir, progress_file(drama_name, getattr(self.args, option_name)))):
                    break
                progress += 1
        return progress

    def get_best_version_dir(self):
        best_version_dir, best_version_name = None, None
        if self.args.load_existing_data != ProjectDramaSR.ExistingDataType.NONE:
            best_progress, last_timestamp = -1, -float("inf")
            for version_name in os.listdir(self.args.workspace_dir):
                if re.match(ProjectDramaSR.VERSION_DIR_PATTERN, version_name) is not None:
                    version_dir = os.path.join(self.args.workspace_dir, version_name)
                    if not self.meta_info_matched(version_dir):
                        continue
                    timestamp, progress = datetime_str2int(version_name[-ProjectDramaSR.VERSION_DIR_TIMESTAMP_LENGTH:]), 0
                    if self.args.load_existing_data == ProjectDramaSR.ExistingDataType.COMPLETE:
                        progress = self.get_project_progress(version_dir)
                    if progress > best_progress or (progress == best_progress and timestamp > last_timestamp):
                        best_version_dir, best_version_name = version_dir, version_name
                        best_progress, last_timestamp = progress, timestamp
        if best_version_dir is None:
            best_version_dir, best_version_name = self.create_new_version_dir()
        os.makedirs(OUTPUT_PATH(best_version_name), exist_ok=True)
        return best_version_dir, best_version_name

    def data_preparation(self):
        for drama_idx in range(self.args.dramas):
            sub_args = SubArguments().get_sub_args(self.args, drama_idx)
            print_step_title("Start data preparation for drama: {:s}".format(sub_args.drama_name), is_start=True)
            # Step 1: Load all drama data (subtitle, face, caption, etc.)
            self.drama_data = DramaData(self, sub_args).load_all_drama_data(ensure_init_seeds=False)
            # Step 2: Perform init seed generation and label propagation
            self.label_propagator = LabelPropagation(self, sub_args)
            self.label_propagator.generate_init_seeds()
            self.label_propagator.perform_label_propagation()
            print_step_title("Finished data preparation for drama: {:s}".format(sub_args.drama_name), is_start=False)

    def data_curation(self):
        if self.args.version_name is None:
            self.args.version_dir, self.args.version_name = self.get_best_version_dir()
        for drama_idx in range(self.args.dramas):
            sub_args = SubArguments().get_sub_args(self.args, drama_idx)
            if not sub_args.drama_flag_sft and not sub_args.drama_flag_rl:
                continue
            print_step_title("Start data curation for drama: {:s}".format(sub_args.drama_name), is_start=True)
            # Step 1: Load all drama data (subtitle, face, caption, etc.)
            self.drama_data = DramaData(self, sub_args).load_all_drama_data()
            # Step 2: Generate SFT and RL data by request
            self.drama_curator = DataCuration(self, sub_args).curate_reasoning_data()
            print_step_title("Finished data curation for drama: {:s}".format(sub_args.drama_name), is_start=False)

    def model_training(self):
        if self.args.version_name is None:
            self.args.version_dir, self.args.version_name = self.get_best_version_dir()
        print_step_title("Start model training.", is_start=True)
        args_list_sft, args_list_rl = [], []
        for drama_idx in range(self.args.dramas):
            sub_args = SubArguments().get_sub_args(self.args, drama_idx)
            if sub_args.drama_flag_sft:
                args_list_sft.append(sub_args)
            if sub_args.drama_flag_rl:
                args_list_rl.append(sub_args)
        self.model_trainer = ModelTraining(self, self.args, args_list_sft, args_list_rl).train_reasoning_model()
        print_step_title("Finished model training", is_start=False)

    def model_inference(self):
        all_cached_models_dir = os.path.join(self.args.root_absolute_path, CACHED_MODELS_PATH)
        for model_timestamp in self.args.trained_model_timestamp.split(","):
            cached_model_dir = os.path.join(all_cached_models_dir, RL_MODEL_BASEDIR(model_timestamp))
            rl_starter_template_file = os.path.join(ASSET_PATH, RL_STARTER_FILE)
            rl_starter_replaced_file = os.path.join(cached_model_dir, RL_STARTER_FILE)
            set_args_from_config_file(rl_starter_template_file, rl_starter_replaced_file, args, overwrite=True, allow_no_key_diff=True)
            all_snapshots_dir = os.path.join(cached_model_dir, "snapshots")
            all_test_results_dir = os.path.join(cached_model_dir, "test_results_{:0.2f}_{:0.2f}".format(self.args.infer_temperature, self.args.infer_top_p))
            os.makedirs(all_test_results_dir, exist_ok=True)
            for snapshot_name in os.listdir(all_snapshots_dir):
                if not snapshot_name.startswith(RL_SNAPSHOT_PREFIX):
                    continue
                self.args.model_path = os.path.join(all_snapshots_dir, snapshot_name)
                self.args.model_name = MODEL_NAME_DICT_LOCAL[self.args.target_model.value]
                test_result_dir = os.path.join(all_test_results_dir, snapshot_name)
                os.makedirs(test_result_dir, exist_ok=True)
                all_finished = True
                for drama_idx in range(self.args.dramas):
                    sub_args = SubArguments().get_sub_args(self.args, drama_idx)
                    if not sub_args.drama_flag_infer:
                        continue
                    infer_data_file = os.path.join(test_result_dir, INFER_DATA_FILE(sub_args.drama_name, sub_args.infer_data_str))
                    if not os.path.exists(infer_data_file):
                        all_finished = False
                        break
                if all_finished:
                    continue
                model_relative_path = os.path.relpath(self.args.model_path, all_cached_models_dir)
                print_step_title("Start inference for model {:s}.".format(model_relative_path), is_start=True)
                ts_starter_template_file = os.path.join(ASSET_PATH, TS_STARTER_FILE)
                ts_starter_replaced_file = os.path.join(test_result_dir, TS_STARTER_FILE)
                replace_template_file(ts_starter_template_file, ts_starter_replaced_file, self.args)
                self.model_inferer = ModelInference(self)
                self.model_inferer.start_server(self.args, ts_starter_replaced_file)
                for drama_idx in range(self.args.dramas):
                    sub_args = SubArguments().get_sub_args(self.args, drama_idx)
                    if not sub_args.drama_flag_infer:
                        continue
                    sub_args.test_result_dir = test_result_dir
                    print_module_title("Start model inference for drama {:s}.".format(sub_args.drama_name), is_start=True)
                    self.drama_data = DramaData(self, sub_args).load_all_drama_data()
                    self.data_curator = DataCuration(self, sub_args).perform_model_inference()
                    print_module_title("Finished model inference for drama {:s}.".format(sub_args.drama_name), is_start=False)
                self.model_inferer.stop_server()
                print_step_title("Finished inference for model {:s}.".format(model_relative_path), is_start=True)

    def result_statistics(self):
        all_cached_models_dir = os.path.join(self.args.root_absolute_path, CACHED_MODELS_PATH)
        for drama_idx in range(args.dramas):
            sub_args = SubArguments().get_sub_args(args, drama_idx)
            if not sub_args.drama_flag_infer:
                continue
            print_step_title("Start model inference for drama {:s}.".format(sub_args.drama_name), is_start=True)
            self.drama_data = DramaData(self, sub_args).load_all_drama_data()
            model_dir_list = []
            for model_timestamp in self.args.trained_model_timestamp.split(","):
                model_dir_list.append(os.path.join(all_cached_models_dir, RL_MODEL_BASEDIR(model_timestamp)))
            # self.data_curator = DataCuration(self, sub_args).get_statistics(model_dir_list)
            self.data_curator = DataCuration(self, sub_args).error_analysis(model_dir_list)
            print_step_title("Finished model inference for drama {:s}.".format(sub_args.drama_name), is_start=False)


def get_default_parser():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--root_absolute_path", type=str, default="/",
                        help="absolute path to the drama data directory")
    parser.add_argument("--drama_data_dir", type=str, default="./drama_data/",
                        help="path to the drama data directory")
    parser.add_argument("--workspace_dir", type=str, default="./workspace/",
                        help="path to the workspace directory")
    # Drama info
    parser.add_argument("--drama_separator", type=str, default=",",
                        help="the separator to split dramas in sft/rl/infer lists")
    parser.add_argument("--drama_name_sft", type=str, default="",
                        help="list of drama names (separated by comma) for generating SFT data")
    parser.add_argument("--drama_name_rl", type=str, default="",
                        help="list of drama names (separated by comma) for generating RL data")
    parser.add_argument("--drama_name_infer", type=str, default="",
                        help="list of drama names (separated by comma) for inference")
    parser.add_argument("--working_stage", type=ProjectDramaSR.WorkingStage,
                        choices=list(ProjectDramaSR.WorkingStage), help="the working stage of the program")
    # Data arguments
    parser.add_argument("--init_seeds_option", type=LabelPropagation.InitSeedsOption, default=LabelPropagation.InitSeedsOption.ONE_PERCENT,
                        choices=list(LabelPropagation.InitSeedsOption), help="option for generating init seeds")
    parser.add_argument("--label_prop_option", type=LabelPropagation.LabelPropOption, default=LabelPropagation.LabelPropOption.AP_WIN,
                        choices=list(LabelPropagation.LabelPropOption), help="option for label propagation")
    parser.add_argument("--sft_data_option", type=DataCuration.SFTDataOption, default=DataCuration.SFTDataOption.GPT5C,
                        choices=list(DataCuration.SFTDataOption), help="option for generating SFT data")
    parser.add_argument("--sft_workers", type=int, default=64,
                        help="number of workers for SFT data collection")
    parser.add_argument("--rl_data_option", type=DataCuration.RLDataOption, default=DataCuration.RLDataOption.NONE,
                        choices=list(DataCuration.RLDataOption), help="option for generating RL data")
    parser.add_argument("--rl_workers", type=int, default=32,
                        help="number of workers for RL data collection")
    parser.add_argument("--api_key", type=str, default="",
                        help="the API key to call proprietary large language models")
    parser.add_argument("--base_url", type=str, default="",
                        help="the base URL to call proprietary large language models")
    parser.add_argument("--enforce_refresh", type=int, default=0,
                        choices=[0, 1], help="whether to enforce executing all procedures regardless of file existence")
    parser.add_argument("--enforce_retrial", type=int, default=1,
                        choices=[0, 1], help="whether to enforce retrying all failure samples (only for SFT data generation)")
    # Training arguments
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="the number of GPUs for training or inference")
    parser.add_argument("--target_model", type=ModelTraining.TargetModel, default=ModelTraining.TargetModel.Q3_8B,
                        choices=list(ModelTraining.TargetModel), help="the target model for reasoning")
    parser.add_argument("--sft_validation_ratio", type=float, default=0.0,
                        help="the ratio of SFT training data withheld for validation")
    parser.add_argument("--sft_enforce_retraining", type=int, default=0,
                        help="whether to enforce SFT retraining even if a model with the same setting exists")
    parser.add_argument("--sft_learning_rate", type=float, default=0.00001,
                        help="the learning rate for SFT")
    parser.add_argument("--sft_num_epochs", type=int, default=5,
                        help="the number of training epochs for SFT")
    parser.add_argument("--rl_validation_ratio", type=float, default=0.1,
                        help="the ratio of RL training data withheld for validation")
    parser.add_argument("--rl_enforce_retraining", type=int, default=0,
                        help="whether to enforce RL retraining even if a model with the same setting exists")
    parser.add_argument("--rl_learning_rate", type=float, default=0.000001,
                        help="the learning rate for RL")
    parser.add_argument("--rl_num_epochs", type=int, default=2,
                        help="the number of training epochs for RL")
    parser.add_argument("--rl_group_size", type=int, default=4,
                        help="the group size (in GRPO) for RL")
    parser.add_argument("--rl_batch_size", type=int, default=12,
                        help="the sample batch size for RL")
    parser.add_argument("--rl_temperature", type=float, default=0.6,
                        help="the temperature of model sampling for RL")
    parser.add_argument("--rl_top_p", type=float, default=0.95,
                        help="the top_p of model sampling for RL")
    parser.add_argument("--rl_presence_penalty", type=float, default=0.1,
                        help="the presence penalty of training the RL model")
    parser.add_argument("--rl_frequency_penalty", type=float, default=0.3,
                        help="the frequency penalty of training the RL model")
    parser.add_argument("--rl_max_turns", type=int, default=15,
                        help="the maximum turns of a trajectory for RL")
    parser.add_argument("--rl_kl_loss_coef", type=float, default=0.0001,
                        help="the KL loss coefficient for RL")
    parser.add_argument("--rl_kl_coef", type=float, default=0.05,
                        help="the KL coefficient for penalizing the reward of RL")
    parser.add_argument("--rl_penalty_length", type=float, default=0.01,
                        help="the penalty coefficient of total tool calls")
    parser.add_argument("--rl_penalty_duplicate", type=float, default=0.1,
                        help="the penalty coefficient of duplicate tool calls")
    # Inference arguments
    parser.add_argument("--conda_env_name", type=str, default=None,
                        help="conda environment name (empty for not calling conda activate $name)")
    parser.add_argument("--trained_model_timestamp", type=str, default="",
                        help="timestamp of the trained model (used to locate model dir)")
    parser.add_argument("--infer_workers", type=int, default=32,
                        help="number of workers for model inference")
    parser.add_argument("--api_key_local", type=str, default="",
                        help="the API key to call local large language models")
    parser.add_argument("--base_port_id", type=int, default=8000,
                        help="the base port ID to call local large language models")
    parser.add_argument("--infer_temperature", type=float, default=0.6,
                        help="the temperature of model sampling for inference")
    parser.add_argument("--infer_top_p", type=float, default=0.95,
                        help="the top_p of model sampling for inference")
    # Version controller
    parser.add_argument("--version_identifier", type=str, default="unnamed_version",
                        help="version identifier")
    parser.add_argument("--load_existing_data", type=ProjectDramaSR.ExistingDataType, default=ProjectDramaSR.ExistingDataType.COMPLETE,
                        choices=list(ProjectDramaSR.ExistingDataType), help="option to load existing data")
    return parser


def parse_arguments(parser):
    args = parser.parse_args()
    args.drama_list_sft = args.drama_name_sft.split(args.drama_separator) if args.drama_name_sft else []
    args.drama_list_rl = args.drama_name_rl.split(args.drama_separator) if args.drama_name_rl else []
    args.drama_list_infer = args.drama_name_infer.split(args.drama_separator) if args.drama_name_infer else []
    args.drama_list = args.drama_list_sft + args.drama_list_rl + args.drama_list_infer
    args.drama_flag_sft = [True] * len(args.drama_list_sft) + [False] * len(args.drama_list_rl) + [False] * len(args.drama_list_infer)
    args.drama_flag_rl = [False] * len(args.drama_list_sft) + [True] * len(args.drama_list_rl) + [False] * len(args.drama_list_infer)
    args.drama_flag_infer = [False] * len(args.drama_list_sft) + [False] * len(args.drama_list_rl) + [True] * len(args.drama_list_infer)
    args.dramas = len(args.drama_list)
    args.language = [DRAMA_INFO[drama_name]["language"] for drama_name in args.drama_list]
    args.episodes = [DRAMA_INFO[drama_name]["episodes"] for drama_name in args.drama_list]
    args.drama_dir = [os.path.join(args.drama_data_dir, drama_name) for drama_name in args.drama_list]
    args.sft_data_str = "{:s}_{:s}_{:s}".format(args.init_seeds_option, args.label_prop_option, args.sft_data_option)
    args.rl_data_str = "{:s}_{:s}_{:s}".format(args.init_seeds_option, args.label_prop_option, args.rl_data_option)
    args.rl_training_data_str = "{:s}_train".format(args.rl_data_str)
    args.rl_validation_data_str = "{:s}_val".format(args.rl_data_str)
    return args


if __name__ == "__main__":
    args = parse_arguments(get_default_parser())
    print(args)
    project = ProjectDramaSR(args)
    if args.working_stage == ProjectDramaSR.WorkingStage.DATA_PREPARATION:
        project.data_preparation()
    elif args.working_stage == ProjectDramaSR.WorkingStage.DATA_CURATION:
        project.data_curation()
    elif args.working_stage == ProjectDramaSR.WorkingStage.MODEL_TRAINING:
        project.model_training()
    elif args.working_stage == ProjectDramaSR.WorkingStage.MODEL_INFERENCE:
        project.model_inference()
    elif args.working_stage == ProjectDramaSR.WorkingStage.RESULT_STATISTICS:
        project.result_statistics()
    else:
        assert False, "Error: unknown working stage."
