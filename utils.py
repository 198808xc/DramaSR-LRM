import os
from datetime import datetime
import re
from enum import Enum


# Part I: Assets and path/dir/file names.
ASSET_PATH = "assets/"
CACHED_MODELS_PATH = "cached_models/"
OUTPUT_PATH = lambda version_name: os.path.join("/cache/DramaSR_output/", version_name)
MODEL_NAME_DICT_PROPRIETARY = {
    "gem3p": "gemini-3.1-pro-preview",
    "gpt5c": "gpt-5-chat-latest",
    "clo45": "claude-opus-4-5-20251101",
}
MODEL_NAME_DICT_LOCAL = {
    "q3_8b": "Qwen3-8B",
}
LLAMA_FACTORY_DIR = "LLaMA-Factory"
DEEPSPEED_CONFIG_FILE = os.path.join(LLAMA_FACTORY_DIR, "examples/deepspeed/ds_z3_config.json")
VERL_TOOL_DIR = "verl-tool"
assert MODEL_NAME_DICT_PROPRIETARY.keys().isdisjoint(MODEL_NAME_DICT_LOCAL.keys())

INIT_SEEDS_FILE = lambda option_str: "init_seeds_{:s}.json".format(str(option_str))
PROP_LABELS_FILE = lambda option_str: "prop_labels_{:s}.json".format(str(option_str))
TOOLSET_INFO_FILE = lambda drama_name, option_str: "toolset_info_{:s}_{:s}.json".format(drama_name, str(option_str))
ALL_DRAMAS_STR = "all"
SFT_DATA_FILE = lambda drama_name, option_str: os.path.join(drama_name, "sft_data_{:s}.json".format(str(option_str)))
SFT_DATA_ALL_FILE = lambda option_str: os.path.join("sft_data_{:s}_{:s}.json".format(ALL_DRAMAS_STR, option_str))
SFT_DATA_INFO_FILE = "dataset_info.json"
SFT_CONFIG_FILE = "sft_config.yaml"
SFT_LOG_FILE = lambda file_part_idx: "training_log_p{:02d}.txt".format(file_part_idx)
SFT_MODEL_PREFIX = "model_sft"
SFT_MODEL_BASEDIR = lambda time_str: "{:s}_{:s}".format(SFT_MODEL_PREFIX, time_str)
RL_DATA_FILE = lambda drama_name, option_str: os.path.join(drama_name, "rl_data_{:s}.json".format(str(option_str)))
RL_DATA_ALL_FILE = lambda option_str: os.path.join("rl_data_{:s}_{:s}.parquet".format(ALL_DRAMAS_STR, option_str))
RL_STARTER_FILE = "rl_starter.sh"
RL_LOG_FILE = lambda file_part_idx: "training_log_p{:02d}.txt".format(file_part_idx)
RL_MODEL_PREFIX = "model_rl"
RL_MODEL_BASEDIR = lambda time_str: "{:s}_{:s}".format(RL_MODEL_PREFIX, time_str)
RL_ROLLOUT_PREFIX = "rollout_data"
RL_ROLLOUT_DIR = lambda output_path, rl_model_dir: os.path.join(output_path, "{:s}_{:s}".format(RL_ROLLOUT_PREFIX, rl_model_dir))
ROLLOUT_TRAINING_DIR = lambda output_path, rl_model_dir: os.path.join(RL_ROLLOUT_DIR(output_path, rl_model_dir), "train")
ROLLOUT_VALIDATION_DIR = lambda output_path, rl_model_dir: os.path.join(RL_ROLLOUT_DIR(output_path, rl_model_dir), "val")
DEFAULT_DATASOURCE_NAME = DEFAULT_TOOLSET_NAME = "speaker_recognition"
RL_SNAPSHOT_PREFIX = "global_step"
RL_SNAPSHOT_DIR = lambda snapshot_idx: "{:s}_{:d}".format(RL_SNAPSHOT_PREFIX, snapshot_idx)
BACKUP_CODE_DIR = "backup_code/"
EXCLUDED_INFO_FILE = "excluded_file_list.txt"
TS_STARTER_FILE = "ts_starter.sh"

INFER_DATA_FILE = lambda drama_name, option_str: "infer_data_{:s}_{:s}.json".format(drama_name, str(option_str))
DATA_SAMPLE_TEMP_FILE = lambda sample_idx: "{:06d}.json".format(sample_idx)
DATA_SAMPLE_TEMP_FILE_PATTERN = r"(\d{6})\.json"
RL_SNAPSHOT_DIR_PATTERN = r"{:s}_(\d+)".format(RL_SNAPSHOT_PREFIX)
INFER_STARTER_FILE = "infer_starter.sh"
CONDA_ENV_COMMAND = lambda conda_env_name: "conda activate {:s}".format(conda_env_name)
LOCAL_PORT_URL = lambda port_id: "http://localhost:{:d}/v1".format(port_id)

TEMPLATE_REPLACEMENT_PATTERN = r"\{\{(.*?)\}\}"
def replace_template_file(template_file, replaced_file, args):
    def replacer(match):
        keyword = match.group(1)
        if hasattr(args, keyword):
            return str(getattr(args, keyword))
        else:
            raise KeyError("Missing variable: {:s}".format(keyword))
    with open(template_file, "r", encoding="utf-8") as file:
        template_content = file.read()
    replaced_content = re.sub(TEMPLATE_REPLACEMENT_PATTERN, replacer, template_content)
    with open(replaced_file, "w", encoding="utf-8") as file:
        file.write(replaced_content)

def match_template_file(template_file, replaced_file, allow_no_key_diff=False):
    with open(template_file, "r", encoding="utf-8") as file:
        template_content = file.read()
    with open(replaced_file, "r", encoding="utf-8") as file:
        replaced_content = file.read()
    template_lines = template_content.splitlines()
    replaced_lines = replaced_content.splitlines()
    if len(template_lines) != len(replaced_lines):
        return None
    keyword_dict = {}
    placeholder_pattern = re.compile(TEMPLATE_REPLACEMENT_PATTERN)
    for t_line, r_line in zip(template_lines, replaced_lines):
        keys = placeholder_pattern.findall(t_line)
        if not keys:
            if allow_no_key_diff or t_line == r_line:
                continue
            else:
                return None
        parts = placeholder_pattern.split(t_line)
        regex_parts = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                regex_parts.append(re.escape(part))
            else:
                regex_parts.append(r"(.*?)")
        line_regex = "^" + "".join(regex_parts) + "$"
        match = re.match(line_regex, r_line)
        if not match:
            return None
        values = match.groups()
        for key, value in zip(keys, values):
            if key in keyword_dict and keyword_dict[key] != value:
                return None
            keyword_dict[key] = value
    return keyword_dict

def set_args_from_config_file(template_file, replaced_file, args, overwrite=False, allow_no_key_diff=False):
    keyword_dict = match_template_file(template_file, replaced_file, allow_no_key_diff=allow_no_key_diff)
    assert keyword_dict is not None, "Error: template_file not matched to replaced_file."
    for key, value in keyword_dict.items():
        if overwrite or not hasattr(args, key):
            setattr(args, key, value)


# Part II: Drama settings and information.
class Language(Enum):
    CHINESE = "CN"
    ENGLISH = "EN"
    def __str__(self):
        return self.value
CHARACTER_NAME_NARRATION = {
    Language.CHINESE: "旁白",
    Language.ENGLISH: "Narration",
}
def OFFSCREEN_KEYWORD_LIST(language: Language):
    TEXT_LIST = [CHARACTER_NAME_NARRATION]
    return [text[language] for text in TEXT_LIST]
NEW_CHARACTER_NAME_PREFIX = {
    Language.CHINESE: "新角色-",
    Language.ENGLISH: "New Character ",
}
NEW_CHARACTER_NAME_FUNC = {
    Language.CHINESE: lambda idx: "新角色-{:d}".format(idx),
    Language.ENGLISH: lambda idx: "New Character {:d}".format(idx),
}
CHARACTER_NAME_OTHERS = {
    Language.CHINESE: "其他",
    Language.ENGLISH: "Other",
}
CHARACTER_NAME_UNKNOWN = {
    Language.CHINESE: "未知",
    Language.ENGLISH: "Unknown",
}
def is_new_character(character, language):
    return character.startswith(NEW_CHARACTER_NAME_PREFIX[language])
def is_others_character(character, language):
    return character == CHARACTER_NAME_OTHERS[language]
def is_unknown_character(character, language):
    return character == CHARACTER_NAME_UNKNOWN[language]
def is_openset_character(character, language):
    return is_others_character(character, language) or is_unknown_character(character, language) or is_new_character(character, language)
def is_invalid_character(character):
    return "!删除!" in character

DRAMA_INFO = {
    "chen_mo_de_zhen_xiang": {"language": Language.CHINESE, "episodes": 12},
    "da_qin_di_guo_zhi_zong_heng": {"language": Language.CHINESE, "episodes": 51},
    "huan_le_song": {"language": Language.CHINESE, "episodes": 42},
    "kuang_biao": {"language": Language.CHINESE, "episodes": 39},
    "ren_shi_jian": {"language": Language.CHINESE, "episodes": 58},
    "san_ti": {"language": Language.CHINESE, "episodes": 30},
    "shan_hai_qing": {"language": Language.CHINESE, "episodes": 23},
    "yi_qi_tong_guo_chuang_1": {"language": Language.CHINESE, "episodes": 34},
    "zhan_chang_sha": {"language": Language.CHINESE, "episodes": 32},
    "zhen_huan_zhuan": {"language": Language.CHINESE, "episodes": 76},
}


# Part III: Print functions.
INDENT_MODULE = 2
INDENT_INFO = 4
INDENT_LOG = 6

def print_with_indent(text: str, indent: int=0, ending=None):
    if ending is None:
        print("{:s}{:s}".format(" " * indent, text))
    elif ending:
        print("{:s}".format(text))
    else:
        print("{:s}{:s}".format(" " * indent, text), end=" ")

def print_step_title(text: str, is_start: bool=True):
    identifier = "=" if is_start else "-"
    side_str = identifier * 20
    print_with_indent("{:s} {:s} {:s}".format(side_str, text, side_str))
    if not is_start:
        print()
    return

def print_module_title(text: str, is_start: bool=True):
    identifier = "=" if is_start else "-"
    side_str = "{:s}{:s}".format(identifier * 5, ">") if is_start else "{:s}{:s}".format("<", identifier * 5)
    print_with_indent("{:s} {:s}".format(side_str, text), indent=INDENT_MODULE)
    return


# Part X: Misc.
TIMESTAMP_PATTERN_LONG = "[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}"
TIMESTAMP_PATTERN_SHORT = "[0-9]{2}:[0-9]{2}:[0-9]{2}"
def time_str2int(time_str: str) -> int:
    if re.match(TIMESTAMP_PATTERN_LONG, time_str):
        return ((int(time_str[0:2]) * 60 + int(time_str[3:5])) * 60 + int(time_str[6:8])) * 1000 + int(time_str[9:12])
    elif re.match(TIMESTAMP_PATTERN_SHORT, time_str):
        return ((int(time_str[0:2]) * 60 + int(time_str[3:5])) * 60 + int(time_str[6:8])) * 1000
    else:
        assert False, "Unknown timestamp format: {:s}".format(time_str)

DATETIME_FORMAT = "%Y%m%d_%H%M%S"
def get_current_datetime() -> str:
    return datetime.now().strftime(DATETIME_FORMAT)

def datetime_str2int(datetime_str: str) -> float:
    return datetime.strptime(datetime_str, DATETIME_FORMAT).timestamp()
