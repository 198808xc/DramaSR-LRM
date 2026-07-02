import json
import os
import re
import ast
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
from decord import VideoReader, cpu
MAX_LENGTH = 1024
# VIDEO_QA_SYSTEM_PROMPT = '''
# You are a video question-answering model. 
# You will be given:
# 1. A video segment description (containing visual and temporal information).
# 2. A question about the video.

# ### Your task:
# 1. Carefully analyze whether the video segment contains information relevant to the question.
# 2. If the video segment provides enough relevant information:
#    - Summarize only the content that helps answer the question.
#    - Do not add details that are not mentioned in the segment.
# 3. If the video segment is irrelevant or does not contain enough information to answer:
#    - Respond: "The video segment does not provide relevant information to answer this question."

# ### Output format:
# Provide your reasoning and summary in the following structure:
# <think>your reasoning</think><answer>your summary</answer>.
# '''
VIDEO_QA_SYSTEM_PROMPT = '''
You are a video question-answering model. You will be given a video segment and a question. Your task is to summarize the information relevant to the question based on the provided video segment.
You must think first before answering. Your thinking process should be in <think></think> tags. Your answer should be in <answer></answer> tags.
If the question is not related to the video, just answer "<answer>The question is not related to the video segment.</answer>".
'''
CAPTION_BASE_PATH = '/opt/huawei/explorer-env/dataset/qjh_train/code/cgbench_code/caption_qwen_wsub_scale_frame'
CAPTION_BASE_PATH_VIDEOMME = '/opt/huawei/explorer-env/dataset/qjh_train/code/videomme_code/caption_qwen_wsub_scale_frame_merged'
# CGBENCH_VIDEO_PATH = '/media/Disk2/Dataset/CG-Bench/cg_videos_720p'
# VIDEOMME_VIDEO_PATH = '/media/Disk3/QiuJihao/long_data/video_mme/data'  
CGBENCH_VIDEO_PATH = '/opt/huawei/explorer-env/dataset/qjh_train/code/cgbench_code/sample_frame/save_frames'
VIDEOMME_VIDEO_PATH = '/opt/huawei/explorer-env/dataset/qjh_train/code/videomme_code/sample_frame/save_frames'

import random
api_key = '1111111'
base_url1 = 'http://127.0.0.1:9081/v1'
base_url2 = 'http://127.0.0.1:9082/v1'

CLIENTS = [OpenAI(
        api_key=api_key,
        base_url=base_url1),
           OpenAI(
        api_key=api_key,
        base_url=base_url2),
          ]

MODEL = "Qwen2.5-VL-32B"


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
        # return func_name, args
    else:
        return None, None, False
    
def extract_and_validate(text):
    """
    提取 <think>、<answer>、<tool> 标签内容，并验证是否符合4种合法格式之一。
    
    参数：
        text (str): 输入字符串
    返回：
        dict: 若格式合法，返回各部分内容的字典；否则返回 {'error': '说明'}
    """
    tags = ['think', 'answer', 'tool']
    results = {}

    # 提取每个标签的内容（只取第一个）
    for tag in tags:
        pattern = fr'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            results[tag] = match.group(1).strip()

    return results


def encode_image_wpath(image_path):
    pil_image = Image.open(image_path)
    w, h = pil_image.size
    # print(f"Original image size: {w}x{h}")
    if w > MAX_LENGTH or h > MAX_LENGTH:
        scale = min(MAX_LENGTH / h, MAX_LENGTH / w)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size)
    # print(f"Resized image size: {pil_image.size[0]}x{pil_image.size[1]}")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def encode_image(image):
    pil_image = Image.fromarray(image)
    w, h = pil_image.size
    # print(f"Original image size: {w}x{h}")
    if w > MAX_LENGTH or h > MAX_LENGTH:
        scale = min(MAX_LENGTH / h, MAX_LENGTH / w)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size)
    # print(f"Resized image size: {pil_image.size[0]}x{pil_image.size[1]}")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
def get_video_duration_cuberoot(path: str) -> float:
    segment_len = 16
    vr = VideoReader(path)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps  # 视频时长（秒）
    num_segments = duration / segment_len
    width = round(num_segments ** (1/3))
    width = max(4, min(8, width))
    return width, num_frames    # 三次根

def process_video_wid(video_path, frame_ids):
    video = VideoReader(video_path, ctx=cpu(0), num_threads=16)
    avg_fps = video.get_avg_fps()
    # frame_list = list(range(0, len(video), interval))
    frame_list = frame_ids
    video = video.get_batch(frame_list).asnumpy()
    video_time = [round(i / avg_fps, 1) for i in frame_list]
    return video, video_time

def sample_video_woid(video):    
    sampled_video = video  
    sampled_video = [encode_image(frame) for frame in sampled_video]
    return sampled_video

def prepare_video_wid(video_path, segment_id):
        # 1. Pre-process video into a hierarchical structure
    print("--- Preparing Video Data ---")
    # video_frames, video_time = process_video(video_path, fps=2)
    width, num_frames = get_video_duration_cuberoot(video_path)
    high_id, medium_id, low_id = segment_id
    sample_frame_num = 16
    # Divide into 4 medium segments
    high_indices = np.array_split(range(num_frames), width)
    high_indices_chunk = high_indices[high_id - 1]
    medium_indices = np.array_split(high_indices_chunk, width)
    medium_indices_chunk = medium_indices[medium_id - 1]
    low_indices = np.array_split(medium_indices_chunk, width)
    selected_low_indices = low_indices[low_id - 1]
    low_frame_ids = np.linspace(selected_low_indices[0], selected_low_indices[-1], sample_frame_num, dtype=int)
    selected_frames, video_time = process_video_wid(video_path, low_frame_ids)
    low_segment_data = {'video': sample_video_woid(selected_frames), 
                        'time': video_time}

    print("Video data prepared successfully.")
    return low_segment_data

def prepare_video_use_image(video_path, segment_id, fps):
    video_path = f'{video_path}/{segment_id[0]}_{segment_id[1]}_{segment_id[2]}'
    image_list = os.listdir(video_path)
    image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
    video = [os.path.join(video_path, frame) for frame in image_list]
    video = [encode_image_wpath(frame) for frame in video]
    frame_ids = [image.split('.')[0] for image in image_list]
    video_time = [round(int(frame_id) / fps, 2) for frame_id in frame_ids]
    low_segment_data = {'video': video, 
                        'time': video_time}
    return low_segment_data

def video_qa_format_message(video, video_time, user_prompt, system_prompt=VIDEO_QA_SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system_prompt}, 
    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    
    for frame_time, frame in zip(video_time, video):
        messages[-1]["content"].append({
            "type": "text",
            "text": f"Video frame at {frame_time} seconds:"
        })
        messages[-1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}"
            }
        })
    return messages


def get_caption_observation(
    caption_path: str,
    segment_id: str,
):
    if not os.path.exists(caption_path):
        return None
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)
    if len(caption_data['captions']) == 3:
        caption_data['captions'].insert(0, [])
    high_data = caption_data['captions'][1]
    medium_data = caption_data['captions'][2]
    low_data = caption_data['captions'][3]
    width = caption_data['width']

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
    
def get_new_caption_observation(
    caption_path: str,
    segment_id: str,
):
    if not os.path.exists(caption_path):
        return None
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)
    if len(caption_data['captions']) == 3:
        caption_data['captions'].insert(0, [])
    high_data = caption_data['captions'][1]
    medium_data = caption_data['captions'][2]
    low_data = caption_data['captions'][3]
    width = caption_data['width']

    try:
        if len(segment_id) == 2:
            high_segment_id, medium_segment_id = map(int, segment_id)
            assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width, "IDs must be in [1, width]"
            caption = medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["caption"]
            frame_time = medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]['frame_time']
            caption = f'Medium-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]})). From {frame_time[0]} to {frame_time[-1]}:{caption}'
            # caption = f'Medium-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]},)):{caption}'
        elif len(segment_id) == 3:
            high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
            assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width and 1 <= low_segment_id <= width, "IDs must be in [1, width]"
            caption = low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["caption"]
            frame_time = low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]['frame_time']
            caption = f'Low-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]},{segment_id[2]})). From {frame_time[0]} to {frame_time[-1]}:{caption}'
        else:
            return None
        return caption
    except Exception as e:
        return None 
def get_video_qa_observation(
    video_path: str,
    segment_id: str,
    query: str,
    width: int
):  
    if not os.path.exists(video_path) or len(segment_id) != 3:
        return None
    try:
        image_list = os.listdir(video_path)
        image_list = sorted(image_list, key=lambda x: int(x.split('_')[0]))
        image_chunk = np.array_split(image_list, width ** 3)
        high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
        image_chunk = image_chunk[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]
        frame_time = [int(img.split('_')[1].split('.')[0]) for img in image_chunk]
        image_chunk = [os.path.join(video_path, img) for img in image_chunk]
        video = [encode_image_wpath(img) for img in image_chunk]
        messages = video_qa_format_message(video, frame_time, query)
        CLIENT = random.choice(CLIENTS)
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
        )
        video_qa_observation = response.choices[0].message.content
        video_qa_parsed = extract_and_validate(video_qa_observation)
        assert "answer" in video_qa_parsed, "Video QA did not return an answer"
        video_qa_answer = video_qa_parsed["answer"]
        video_qa_answer = f'<information>{video_qa_answer}</information>'
        return video_qa_answer
    except Exception as e:
        return None


def get_new_video_qa_observation(
    video_path: str,
    segment_id: str,
    query: str,
    width: int,
    fps: float,
    use_image: bool
):  
    if not os.path.exists(video_path) or len(segment_id) != 3:
        return None
    try:
        if use_image:
            low_segment_data = prepare_video_use_image(video_path, segment_id, fps)
        else:
            low_segment_data = prepare_video_wid(video_path, segment_id)
        messages = video_qa_format_message(low_segment_data['video'], low_segment_data['time'], query)
        CLIENT = random.choice(CLIENTS)
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
        )
        video_qa_observation = response.choices[0].message.content
        video_qa_parsed = extract_and_validate(video_qa_observation)
        assert "answer" in video_qa_parsed, "Video QA did not return an answer"
        video_qa_answer = video_qa_parsed["answer"]
        video_qa_answer = f'<information>{video_qa_answer}</information>'
        return video_qa_answer
    except Exception as e:
        return None

def get_tool_observation(
    parsed_tool_calls: str,
    video_uid: str,
    data_source: str,
    width: int,
    fps: float
):  
    use_image = True
    func_name, args, arg_legal = parse_function_call(parsed_tool_calls)
    if not arg_legal or len(args) > 3:
        return None
    if func_name == 'get_caption':
        if data_source == 'videocaption_cgbench':
            caption_path = f'{CAPTION_BASE_PATH}/{video_uid}.json'
        elif data_source == 'videocaption_videomme':
            caption_path = f'{CAPTION_BASE_PATH_VIDEOMME}/{video_uid}.json'
        else:
            return None
        segment_id = args
        caption = get_new_caption_observation(caption_path, segment_id)
        return caption
    elif func_name == 'video_qa':
        if data_source == 'videocaption_cgbench':
            video_path = f'{CGBENCH_VIDEO_PATH}/{video_uid}.mp4' if not use_image else f'{CGBENCH_VIDEO_PATH}/{video_uid}'
        elif data_source == 'videocaption_videomme':
            video_path = f'{VIDEOMME_VIDEO_PATH}/{video_uid}.mp4' if not use_image else f'{VIDEOMME_VIDEO_PATH}/{video_uid}'
        else:
            return None
        segment_id, query = args
        video_qa_observation = get_new_video_qa_observation(video_path, segment_id, query, width, fps, use_image)
        return video_qa_observation
    else:
        return None
        



