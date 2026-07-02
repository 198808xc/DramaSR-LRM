import os
import sys
import re
import random
from enum import Enum

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from llm_prompts import get_prompt_func
from llm_prompts import ContentErrorType


class Toolset:

    SUPPORTED_TOOLS = {
        "audio_sim": {"parameters": 0},
        "char_relation": {"parameters": 0},
        "video_cap_brief": {"parameters": 0},
        "video_cap_detailed": {"parameters": 0},
    }
    CONTENT_PATTERN = r"<(think|tool|answer)>(.*?)</\1>"
    PARAMETERIZED_TOOL_PATTERN = r"(\w+)\((.*?)\)"

    def __init__(self, language, drama_data, label_set):
        self.language = language
        self.subtitle_data = drama_data.subtitle_data
        self.total_subtitles = self.subtitle_data["subtitles"]
        self.known_character_list = drama_data.known_character_list
        self.relation_data = drama_data.relation_data
        self.caption_data = drama_data.caption_data
        self.label_set = label_set
        self.character_name_others = CHARACTER_NAME_OTHERS[self.language]
        self.character_name_unknown = CHARACTER_NAME_UNKNOWN[self.language]

    def get_subtitle_info_str(self, subtitle_idx, context_length):
        subtitle_str_list, subtitles = [], 0
        prev_episode_idx, curr_subtitle_idx = 0, max(0, subtitle_idx - context_length) - 1
        while curr_subtitle_idx >= 0:
            subtitle_object = self.subtitle_data["subtitle_list"][curr_subtitle_idx]
            if not subtitle_object["is_invalid"]:
                prev_episode_idx = subtitle_object["episode_idx"]
                break
            curr_subtitle_idx -= 1
        for curr_subtitle_idx in range(max(0, subtitle_idx - context_length), min(subtitle_idx + context_length + 1, self.total_subtitles)):
            subtitle_object = self.subtitle_data["subtitle_list"][curr_subtitle_idx]
            if subtitle_object["is_invalid"]:
                continue
            if subtitle_object["episode_idx"] > prev_episode_idx:
                subtitle_str_list.append(get_prompt_func(self.language, "episode_border_prompt"))
                prev_episode_idx = subtitle_object["episode_idx"]
            subtitles += 1
            pseudo_label = self.label_set.get_name_by_code(self.label_set.utterance_label[curr_subtitle_idx]["code"])
            if curr_subtitle_idx == subtitle_idx:
                pseudo_label = self.character_name_unknown
                target_subtitle_idx = subtitles
            subtitle_str_list.append("[{:d}] {:s}:{:s}".format(subtitles, pseudo_label, subtitle_object["text"]))
        return get_prompt_func(self.language, "subtitle_info_str")("\n".join(subtitle_str_list)), target_subtitle_idx

    def get_candidate_str_list(self, subtitle_idx, numbered=False):
        rel_info = self.label_set.utterance_label[subtitle_idx]["rel_info"]
        candidate_str_list, others_in_candidates = [], False
        for _, candidate_name in enumerate(rel_info["_list"]):
            candidate_str_list.append("{:s}".format(candidate_name))
            if candidate_name == self.character_name_others:
                others_in_candidates = True
        if not others_in_candidates:
            candidate_str_list.append("{:s}".format(self.character_name_others))
        if numbered:
            for candidate_idx, candidate_str in enumerate(candidate_str_list):
                candidate_str_list[candidate_idx] = "{:d}.{:s}".format(candidate_idx + 1, candidate_str)
        return candidate_str_list

    def get_candidate_info_str(self, subtitle_idx):
        return get_prompt_func(self.language, "candidate_info_str")("\n".join(self.get_candidate_str_list(subtitle_idx, numbered=True)))

    def get_task_prompt_str(self, subtitle_idx, cheating_answer=None):
        task_prompt_str = get_prompt_func(self.language, "task_prompt_str")(subtitle_idx)
        if cheating_answer is not None:
            task_prompt_str += get_prompt_func(self.language, "cheating_prompt_str")(cheating_answer)
        return task_prompt_str

    def get_audio_sim_str(self, subtitle_idx, *args, **kwargs):
        rel_info = self.label_set.utterance_label[subtitle_idx]["rel_info"]
        ## TODO: test **kwargs for perturbation!
        perturbation = kwargs["perturbation"]
        if len(rel_info["_list"]) == 1:
            perturbation = 0.
        audio_sim_str_list = []
        for candidate_idx in range(len(rel_info["_list"])):
            candidate_name = rel_info["_list"][candidate_idx]
            candidate_sim = rel_info["_dict"][candidate_name]
            candidate_sim += (-perturbation if candidate_idx == 0 else (perturbation if candidate_idx == 1 else 0.))
            audio_sim_str_list.append("{:s}:{:0.4f}".format(candidate_name, candidate_sim))
        return get_prompt_func(self.language, "audio_sim_str")("\n".join(audio_sim_str_list))

    def get_char_relation_str(self, subtitle_idx, *args, **kwargs):
        rel_info = self.label_set.utterance_label[subtitle_idx]["rel_info"]
        char_relation_list = []
        for candidate_idx in range(len(rel_info["_list"])):
            candidate_name = rel_info["_list"][candidate_idx]
            if candidate_name in self.relation_data["character_name_mapping"]:
                candidate_name = self.relation_data["character_name_mapping"][candidate_name]
            if candidate_name is None:
                continue
            if candidate_name in self.relation_data["character_dict"]:
                char_relation_list.extend(self.relation_data["character_dict"][candidate_name])
        if len(char_relation_list) == 0:
            return get_prompt_func(self.language, "char_relation_failure_str")
        char_relation_str_list = [get_prompt_func(self.language, "single_char_relation_str")(*char_relation).strip() for char_relation in char_relation_list]
        return get_prompt_func(self.language, "char_relation_str")("\n".join(char_relation_str_list))

    def locate_caption(self, caption_list, subtitle_idx):
        subtitle_start_timestamp = self.subtitle_data["subtitle_list"][subtitle_idx]["abs_start_timestamp"]
        subtitle_end_timestamp = self.subtitle_data["subtitle_list"][subtitle_idx]["abs_end_timestamp"]
        caption_idx_left, caption_idx_right = 0, len(caption_list) - 1
        caption_idx_start = -1
        while caption_idx_left <= caption_idx_right:
            caption_idx_middle = (caption_idx_left + caption_idx_right) // 2
            if subtitle_start_timestamp >= caption_list[caption_idx_middle]["abs_end_timestamp"]:
                caption_idx_left = caption_idx_middle + 1
                caption_idx_start = caption_idx_middle
            else:
                caption_idx_right = caption_idx_middle - 1
        caption_idx = caption_idx_left
        if caption_idx_start == -1:
            caption_idx_start = 0
        caption_idx_best, score_best = -1, 0
        while True:
            if subtitle_start_timestamp >= caption_list[caption_idx_start]["abs_end_timestamp"]:
                score = caption_list[caption_idx_start]["abs_end_timestamp"] - subtitle_start_timestamp
            elif subtitle_end_timestamp <= caption_list[caption_idx_start]["abs_start_timestamp"]:
                score = subtitle_end_timestamp - caption_list[caption_idx_start]["abs_start_timestamp"]
            else:
                overlap_start_timestamp = max(caption_list[caption_idx_start]["abs_start_timestamp"], subtitle_start_timestamp)
                overlap_end_timestamp = min(caption_list[caption_idx_start]["abs_end_timestamp"], subtitle_end_timestamp)
                score = overlap_end_timestamp - overlap_start_timestamp
            if caption_idx_best == -1 or score > score_best:
                caption_idx_best, score_best = caption_idx_start, score
            if subtitle_end_timestamp <= caption_list[caption_idx_start]["abs_start_timestamp"]:
                break
            caption_idx_start += 1
            if caption_idx_start == len(caption_list):
                break
        return caption_idx_best

    def get_video_cap_brief_str(self, subtitle_idx, *args, **kwargs):
        caption_idx = self.locate_caption(self.caption_data["brief_caption_list"], subtitle_idx)
        video_cap_str = self.caption_data["brief_caption_list"][caption_idx]["description"]
        return get_prompt_func(self.language, "video_cap_brief_str")(video_cap_str)

    def get_video_cap_detailed_str(self, subtitle_idx, *args, **kwargs):
        caption_idx = self.locate_caption(self.caption_data["detailed_caption_list"], subtitle_idx)
        video_cap_str = self.caption_data["detailed_caption_list"][caption_idx]["description"]
        return get_prompt_func(self.language, "video_cap_detailed_str")(video_cap_str)

    def get_system_message(self, prompt):
        return get_prompt_func(self.language, "message")("system", prompt)

    def get_user_message(self, prompt):
        return get_prompt_func(self.language, "message")("user", prompt)

    def get_assistant_message(self, prompt):
        return get_prompt_func(self.language, "message")("assistant", prompt)

    def get_initial_message(self, subtitle_idx, context_length, cheating_answer=None):
        subtitle_info_str, target_subtitle_idx = self.get_subtitle_info_str(subtitle_idx, context_length)
        candidate_info_str = self.get_candidate_info_str(subtitle_idx)
        task_prompt_str = self.get_task_prompt_str(target_subtitle_idx, cheating_answer=cheating_answer)
        system_prompt = get_prompt_func(self.language, "system_prompt")
        user_prompt = get_prompt_func(self.language, "user_prompt")(subtitle_info_str, candidate_info_str, task_prompt_str)
        return [self.get_system_message(system_prompt), self.get_user_message(user_prompt)]

    def get_error_prompt_str(self, error_type, parameter1=None, parameter2=None):
        error_info = get_prompt_func(self.language, "error_info")[error_type]
        if parameter2 is not None:
            error_info = error_info(parameter1, parameter2)
        elif parameter1 is not None:
            error_info = error_info(parameter1)
        return get_prompt_func(self.language, "error_prompt")(error_info)

    def get_tool_call_str(self, tool_name, subtitle_idx, parameter_list, in_rl_training):
        func = getattr(self, "get_{:s}_str".format(tool_name))
        kwargs = {
            "perturbation": (0.0 * random.randint(0, 1)) if in_rl_training else 0.0
        }
        return func(subtitle_idx, *parameter_list, **kwargs)

    def get_called_tool_info_message(self, called_tool_info_str):
        return get_prompt_func(self.language, "called_tool_info")(called_tool_info_str)

    @staticmethod
    def is_tool_called(tool_info, tool_info_list):
        for called_tool_info in tool_info_list:
            if tool_info["name"] == called_tool_info["name"] and tool_info["parameter_list"] == called_tool_info["parameter_list"]:
                return True
        return False

    @staticmethod
    def decode_content(content):
        content_matches = re.finditer(Toolset.CONTENT_PATTERN, content, re.DOTALL)
        if not content_matches:
            return False, None, None
        decoded_content = {}
        for content_match in content_matches:
            key, value = content_match.groups()
            if key in decoded_content:
                return True, key, None
            decoded_content.update({key: value.strip()})
        return True, None, decoded_content

    def get_next_user_prompt(self, subtitle_idx, content, message_info, in_rl_training=False):
        decode_success, duplicate_key, decoded_content = Toolset.decode_content(content)
        finished, valid, tool_info, user_prompt = False, False, None, None
        if not decode_success:
            user_prompt = self.get_error_prompt_str(ContentErrorType.PATTERN_NOT_MATCHED)
        elif duplicate_key is not None:
            user_prompt = self.get_error_prompt_str(ContentErrorType.MULTIPLE_IDENTIFIER, duplicate_key)
        elif not "think" in decoded_content:
            user_prompt = self.get_error_prompt_str(ContentErrorType.NO_THINK_IDENTIFIER)
        elif ("tool" in decoded_content) == ("answer" in decoded_content):
            user_prompt = self.get_error_prompt_str(ContentErrorType.TOOL_AND_ANSWER)
        elif "tool" in decoded_content:
            tool_match = re.match(Toolset.PARAMETERIZED_TOOL_PATTERN, decoded_content["tool"])
            tool_name, parameter_list = decoded_content["tool"], []
            if tool_match:
                tool_name, parameter_list_str = tool_match.groups()
                parameter_list = [parameter.strip() for parameter in parameter_list_str.split(',')]
            if not tool_name in Toolset.SUPPORTED_TOOLS:
                user_prompt = self.get_error_prompt_str(ContentErrorType.UNSUPPORTED_TOOL, tool_name)
            elif len(parameter_list) != Toolset.SUPPORTED_TOOLS[tool_name]["parameters"]:
                user_prompt = self.get_error_prompt_str(ContentErrorType.INVALID_PARAMETER_LIST, tool_name)
            else:
                tool_info = {"name": tool_name, "parameter_list": parameter_list}
                if Toolset.is_tool_called(tool_info, message_info["tool_info_list"]):
                    tool_func_str = "{:s}({:s})".format(tool_info["name"], ",".join(tool_info["parameter_list"]))
                    user_prompt = self.get_error_prompt_str(ContentErrorType.DUPLICATE_TOOL_CALL, tool_func_str)
                    tool_info = None
                else:
                    valid = True
                    user_prompt = self.get_tool_call_str(tool_name, subtitle_idx, parameter_list, in_rl_training)
                    message_info["tool_info_list"].append(tool_info)
        else:
            must_called_tool_list = [{"name": "audio_sim", "parameter_list": []}]
            if message_info["cheating"]:
                must_called_tool_list.append({"name": "video_cap_detailed", "parameter_list": []})
            for must_called_tool in must_called_tool_list:
                if not Toolset.is_tool_called(must_called_tool, message_info["tool_info_list"]):
                    user_prompt = self.get_error_prompt_str(ContentErrorType.TOOL_NOT_CALLED_YET, must_called_tool["name"])
                    break
            if user_prompt is None:
                character_name = decoded_content["answer"]
                if character_name in self.get_candidate_str_list(subtitle_idx):
                    finished, valid = True, True
                    user_prompt = character_name
                else:
                    user_prompt = self.get_error_prompt_str(ContentErrorType.ANSWER_IS_NOT_CANDIDATE, character_name)
        if not finished:
            called_tool_info_list = ["{:s}({:s})".format(called_tool_info["name"], ",".join(called_tool_info["parameter_list"]))
                                     for called_tool_info in message_info["tool_info_list"]]
            called_tool_info_str = "\n".join(called_tool_info_list)
            user_prompt += "\n{:s}".format(self.get_called_tool_info_message(called_tool_info_str))
        return finished, valid, tool_info, user_prompt
