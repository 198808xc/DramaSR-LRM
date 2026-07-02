# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os
import tempfile
import json
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
# no video qa
# VIDEO_TREE_CAPTION_SYSTEM_PROMPT = "\n## INSTRUCTIONS\nThe video is divided into four hierarchical levels: Most-coarse, High-level, Medium-level, and Low-level. Answer the given question. You must conduct reasoning inside <think> and </think> first every time before you get new information. After reasoning, if you find you lack some knowledge, you can call tool `get_caption((high_segment_id, medium_segment_id, low_segment_id))` to get 1 segments caption(`high_segment_id, medium_segment_id, low_segment_id` is from 1 to {width}). (high_segment_id, medium_segment_id, low_segment_id) is a triplet, when you get high_level_caption, the triplet should be `(high_segment_id,)`, when you get medium_level_caption, the triplet should be `(high_segment_id, medium_segment_id,)`and it will return the segment's caption. If you find no further external knowledge needed, you can provide the answer inside <answer> and </answer> after another thinking.\n"
# with video qa
VIDEO_TREE_CAPTION_SYSTEM_PROMPT = "\n## INSTRUCTIONS\nThe video is divided into four hierarchical levels: Most-coarse, High-level, Medium-level, and Low-level. Answer the given question. You must conduct reasoning inside <think> and </think> first every time before you get new information. After reasoning, if you find you lack some knowledge, you can call tool `get_caption((high_segment_id, medium_segment_id, low_segment_id))` to get 1 segments caption(`high_segment_id, medium_segment_id, low_segment_id` is from 1 to {width}). (high_segment_id, medium_segment_id, low_segment_id) is a triplet, when you get high_level_caption, the triplet should be `(high_segment_id,)`, when you get medium_level_caption, the triplet should be `(high_segment_id, medium_segment_id,)`and it will return the segment's caption. If you think the question's answer can be found in some segments, but the caption is not detailed enough, use the tool `video_qa((high_segment_id, medium_segment_id, low_segment_id), query)` to get the relevant information of the query from the video segment. This tool can only be used after you have requested the low-level caption. If you find no further external knowledge needed, you can provide the answer inside <answer> and </answer> after another thinking.\n"


DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
SEARCH_R1_CONTENT_PREFIX = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question")
    option = row.get("choices")
    right_answer = row.get("right_answer")
    video_uid = row.get("video_uid")
    clue_intervals = row.get("clue_intervals")
    width = row.get('width')
    duration = row.get('duration')
    
    option = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(option)]
    option = '\n'.join(option)
    user_content = f"{question}\n{option}\n"

    caption_path = f'{args.caption_dir}/{video_uid}.json'
    with open(caption_path, 'r') as f:
        caption = json.load(f)
    
    most_coarse_data = caption['captions'][0][0]
    video_time = most_coarse_data['frame_time']
    video_caption = most_coarse_data['caption']
    time_prompt = f"The total duration of the video is {video_time[-1]} seconds."
    caption_prompt = f"Most-coarse Caption:{video_caption}"

    # Build prompt structure

    prompt = [{"role": "system", "content": VIDEO_TREE_CAPTION_SYSTEM_PROMPT.format(width=width)}, {"role": "user", "content": user_content},{"role": "user", "content": time_prompt},{"role": "user", "content": caption_prompt}]

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = {
            'style': 'rule',
            'ground_truth': right_answer
        }
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # Process data source
    data_source_tagged = "videocaption_cgbench"

    # Build tools kwargs structure
    # tools_kwargs = {
    #     "search": {
    #         "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
    #     }
    # }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "video_uid": video_uid,
        "split": current_split_name,
        "clue_intervals": clue_intervals,
        "width": width,
        "duration": duration,
        "data_source": data_source_tagged,
        # "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": 'get_caption',
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    

    # Download and process files using temporary directory

    # for split in ["sample_300_train", "sample_300_val"]:
    for split in ["sample_300"]:
        parquet_filename = f"{split}.parquet"
        logger.info(f"Processing {split} split...")
        local_parquet_filepath = f'{args.data_dir}/{split}.parquet'
        # Load and process Parquet file
        df_raw = pd.read_parquet(local_parquet_filepath)
        logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

        def apply_process_row(row, split_name=split):
            return process_single_row(row, current_split_name=split_name, row_index=row.name)

        df_processed = df_raw.apply(apply_process_row, axis=1)

        # Save processed DataFrame
        output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
        df_processed.to_parquet(output_file_path, index=False)
        logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
        processed_files.append(output_file_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--data_dir", default="/home/QiuJihao/Workspace/RL/latest_code/qwen3_code/verl_tool/examples/data_preprocess", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir",
        default="/home/QiuJihao/Workspace/RL/latest_code/qwen3_code/train_data/train_data_wvideoqa",
        help="Local directory to save the processed Parquet files.",
    )

    parser.add_argument(
        "--caption_dir",
        default="/home/QiuJihao/Workspace/RL/latest_code/private_model/cg_bench_code/caption",
        help="Local directory to save the processed Parquet files.",
    )


    args = parser.parse_args()

    # System and user content configuration
    main()
