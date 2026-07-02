import json
import os
import re
import ast

CAPTION_BASE_PATH = '/home/QiuJihao/Workspace/RL/latest_code/private_model/cg_bench_code/caption'
CAPTION_BASE_PATH_VIDEOMME = '/home/QiuJihao/Workspace/RL/latest_code/private_model/videomme_code/caption_qwen'

def parse_function_call(call_str):
    """
    解析形如 function_name((9,1,2)) 的字符串
    返回函数名和参数（转为 Python 对象）
    """
    func_pattern = r"(\w+)\((.*)\)"
    match = re.match(func_pattern, call_str.strip())
    arg_legal = False
    if match:
        func_name = match.group(1)
        arg_str = match.group(2).strip()
        try:
            # 直接解析
            args = ast.literal_eval(arg_str) if arg_str else ()
            arg_legal = True
        except Exception as e:
            print(f"参数解析失败: {arg_str}，错误: {e}")
            args = arg_str  # 保留原字符串
            arg_legal = False
        return func_name, args, arg_legal
    else:
        return None, None, False
def get_caption_observation(
    parsed_tool_calls: str,
    video_uid: str,
    data_source: str,
):
    func_name, args, arg_legal = parse_function_call(parsed_tool_calls)
    if not arg_legal or func_name != 'get_caption' or len(args) > 3:
        return None
    if data_source == 'videocaption_cgbench':
        caption_path = f'{CAPTION_BASE_PATH}/{video_uid}.json'
    elif data_source == 'videocaption_videomme':
        caption_path = f'{CAPTION_BASE_PATH_VIDEOMME}/{video_uid}.json'
    else:
        return None
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)
    
    high_data = caption_data['captions'][1]
    medium_data = caption_data['captions'][2]
    low_data = caption_data['captions'][3]
    width = caption_data['width']

    segment_id = args

    try:
        if len(segment_id) == 1:
            high_segment_id = int(segment_id[0])
            assert 1 <= high_segment_id <= width, "ID must be in [1, width]"
            caption = high_data[high_segment_id - 1]["caption"]
            caption = f'High-level Caption from tool get_caption(({segment_id[0]},)):{caption}'
        elif len(segment_id) == 2:
            high_segment_id, medium_segment_id = map(int, segment_id)
            assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width, "IDs must be in [1, width]"
            caption = medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["caption"]
            caption = f'Medium-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]},)):{caption}'
        elif len(segment_id) == 3:
            high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
            assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width and 1 <= low_segment_id <= width, "IDs must be in [1, width]"
            caption = low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["caption"]
            caption = f'High-level Caption from tool get_caption(({segment_id[0]},)):{caption}'
        
        return caption
    except Exception as e:
        return None


