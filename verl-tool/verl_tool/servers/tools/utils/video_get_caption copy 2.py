import json
import os
import re
import ast
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
MAX_LENGTH = 512
VIDEO_QA_SYSTEM_PROMPT = '''
You are a video question-answering model. 
You will be given:
1. A video segment description (containing visual and temporal information).
2. A question about the video.

### Your task:
1. Carefully analyze whether the video segment contains information relevant to the question.
2. If the video segment provides enough relevant information:
   - Summarize only the content that helps answer the question.
   - Do not add details that are not mentioned in the segment.
3. If the video segment is irrelevant or does not contain enough information to answer:
   - Respond: "The video segment does not provide relevant information to answer this question."

### Output format:
Provide your reasoning and summary in the following structure:
<think>your reasoning</think><answer>your summary</answer>.
'''
CAPTION_BASE_PATH = '/home/QiuJihao/Workspace/RL/latest_code/private_model/cg_bench_code/caption'
CAPTION_BASE_PATH_VIDEOMME = '/home/QiuJihao/Workspace/RL/latest_code/private_model/videomme_code/caption_qwen_wsub'
CGBENCH_VIDEO_PATH = '/home/QiuJihao/Workspace/RL/latest_code/private_model/cg_bench_code/save_frames'
VIDEOMME_VIDEO_PATH = '/home/QiuJihao/Workspace/RL/latest_code/private_model/videomme_code/save_frames'  



api_key = '1111111'
base_url = 'http://127.0.0.1:9082/v1'
CLIENT = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

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


def encode_image(image_path):
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
        video = [encode_image(img) for img in image_chunk]
        messages = video_qa_format_message(video, frame_time, query)
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
    width: int
):
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
        caption = get_caption_observation(caption_path, segment_id)
        return caption
    elif func_name == 'video_qa':
        if data_source == 'videocaption_cgbench':
            video_path = f'{CGBENCH_VIDEO_PATH}/{video_uid}'
        elif data_source == 'videocaption_videomme':
            video_path = f'{VIDEOMME_VIDEO_PATH}/{video_uid}'
        else:
            return None
        segment_id, query = args
        video_qa_observation = get_video_qa_observation(video_path, segment_id, query, width)
        return video_qa_observation
    else:
        return None
        



