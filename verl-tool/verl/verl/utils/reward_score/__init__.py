# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated
import os
import re

penalty_length = float(os.environ.get("penalty_length"))
penalty_duplicate = float(os.environ.get("penalty_duplicate"))

def parse_speaker_recognition_conversation(structured_text):
    parts = re.split(r"^\s*(user|assistant)\s*$", structured_text, flags=re.MULTILINE)
    conversation = []
    current_role = "assistant"
    for part in parts:
        part_stripped = part.strip()
        if part_stripped in ["user", "assistant"]:
            current_role = part_stripped
        elif part_stripped:
            if current_role == "assistant":
                pattern = r"<(.*?)>(.*?)</\1>"
                matches = re.findall(pattern, part, flags=re.DOTALL)
                units = [{"identifier": identifier, "text": text} for identifier, text in matches]
                if units:
                    conversation.append({
                        "role": current_role,
                        "content": units
                    })
            elif current_role == "user":
                conversation.append({
                    "role": current_role,
                    "text": part_stripped
                })
    return conversation

def speaker_recognition_reward(conversation, ground_truth):
    rounds = 0
    called_tool_set, duplicate_calls = set(), 0
    answer_list = []
    for item in conversation:
        if item["role"] == "assistant":
            for unit in item["content"]:
                if unit["identifier"] == "think":
                    ...
                elif unit["identifier"] == "tool":
                    tool_name = unit["text"]
                    if tool_name in called_tool_set:
                        duplicate_calls += 1
                    else:
                        called_tool_set.add(tool_name)
                elif unit["identifier"] == "answer":
                    answer_list.append(unit["text"])
                    break
                else:
                    return -1
            rounds += 1
    if len(answer_list) != 1:
        return -1
    else:
        answer = answer_list[0]
        reward = (answer == ground_truth["character_gt"]) or \
            (not answer in ground_truth["candidate_list"] and \
             ground_truth["character_gt"] == ground_truth["name_others"])
        return max(0, reward - penalty_length * max(0, rounds - 2) - penalty_duplicate * duplicate_calls)

def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    elif data_source == "speaker_recognition":
        conversation = parse_speaker_recognition_conversation(solution_str)
        res = speaker_recognition_reward(conversation, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
