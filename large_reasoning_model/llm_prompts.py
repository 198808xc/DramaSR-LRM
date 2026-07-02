import os
import sys
from enum import Enum

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import Language


PROMPT_SEPARATOR = "=" * 20
INFO_SEPARATOR = "-" * 10

SUBTITLE_INFO_STR_CN = lambda subtitle_info_str: f'''
## 上下文台词内容如下：
{subtitle_info_str}
{PROMPT_SEPARATOR}
注：每行表示一句台词，格式为"[序号] 说话人:台词"。例如"[1] 小明:你好"。如果说话人为"未知"，则代表此处的说话人身份暂时无法确定；如果说话人为"新角色-X"（其中X为任意数字编号）或者"其他"，则表示说话人是除了被命名的候选人外的其他角色，其中"新角色-X"代表通过声纹特征聚类得到的未命名角色（相同编号代表相同角色，不同编号代表不同角色），而"其他"则表示声纹特征未被聚类的未命名角色。特别强调：所有台词的说话人标签都来源于声纹相似度计算和模型推理，不一定是正确答案！
'''
SUBTITLE_INFO_STR_EN = lambda subtitle_info_str: f'''
'''

EPISODE_BORDER_PROMPT_CN = f"{INFO_SEPARATOR}这是两集之间的分界线{INFO_SEPARATOR}"
EPISODE_BORDER_PROMPT_EN = ""

CANDIDATE_INFO_STR_CN = lambda candidate_info_str: f'''
## 说话人候选名单如下：
{candidate_info_str}
{PROMPT_SEPARATOR}
注：候选名单中，"新角色-X"和"其他"的含义同上。
'''
CANDIDATE_INFO_STR_EN = lambda candidate_info_str: f'''
'''

TASK_PROMPT_STR_CN = lambda subtitle_idx: f'''
## 你需要判断说话人的台词序号为：[{subtitle_idx}]。
## 补充说明：
1. 判定标准：如果该台词的正确说话人确实是某个未命名角色，那么你输出"新角色-X"和"其他"都会被判定为正确，输出任何已知角色都会被判定为错误；如果该台词的正确说话人是某个已知角色，那么你只有准确输出该角色的名字才会被判定为正确，输出任何其他内容都会被判定为错误。
2. "新角色-X"和"其他"可能包含两种情况。其一，这可能意味着说话人是影视剧中临时出现的未命名角色，如广播、警察、路人、工作人员等；其二，这可能是某个已知角色的语音或者画外音，但由于在该角色的人脸特征在相应片段中未出现或者难以识别、该角色声纹特征发生显著变化等原因，算法未将其归类为正确的角色。你需要根据上下文和调用工具返回的信息进行推理，判断出这些角色的正确标签！
'''
TASK_PROMPT_STR_EN = lambda subtitle_idx: f'''
'''
CHEATING_PROMPT_STR_CN = lambda cheating_answer: f'''
## 注意：上述台词的正确说话人是：{cheating_answer}。请你提供一个合理的工具调用和推理过程，确保得到正确的答案。如果你坚持认为正确答案是"新角色-X"或者"其他"（与正确答案不同），那么请积极考虑以下可能性：你认为的"新角色-X"或者"其他"事实上就是{cheating_answer}，只不过因为其人脸特征不明显或者其声纹特征发生显著变化，算法暂时将其归类为"新角色-X"或者"其他"。此时，请你务必调用video_cap_detailed工具，以获得画面细节描述；必要时，还应当调用video_cap_brief工具以获得短期故事背景，以及调用char_relation工具以获得人物关系——这些信息通常有助于判断说话人。你需要结合这些信息，通过合理的推理指出这个事实，并且最终给出正确的答案。
## 总结&再次强调：
最终的回答必须以"<answer>{cheating_answer}</answer>"结束，但在推理过程中不能提及已经知道答案的事实！
最终的回答必须以"<answer>{cheating_answer}</answer>"结束，但在推理过程中不能提及已经知道答案的事实！
最终的回答必须以"<answer>{cheating_answer}</answer>"结束，但在推理过程中不能提及已经知道答案的事实！
'''
CHEATING_PROMPT_STR_EN = lambda cheating_answer: f'''
'''

AUDIO_SIM_STR_CN = lambda audio_sim_list_str: f'''
## 候选人音频相似度如下：
{audio_sim_list_str}
{PROMPT_SEPARATOR}
注：每行表示一位候选人的音频相似度信息，格式为"说话人:相似度"，如"小明:0.4835"。其中相似度是一个0-1之间的浮点数，精确到小数点后4位，表示角色语音库与当前台词的语音相似度。信息中包含除"其他"选项外所有真实候选人，包括根据声纹聚类得到的未命名角色（以"新角色-X"表示）。
''' if len(audio_sim_list_str) > 0 else f'''
未查询到任何音频相似度信息。这意味着没有任何候选项，即正确答案应为"其他"。
'''
# AUDIO_SIM_STR_CN = lambda audio_sim_list_str: f'''
# 未查询到任何音频相似度信息，请根据上下文判断说话人。
# ''' if len(audio_sim_list_str) > 0 else f'''
# 未查询到任何音频相似度信息，请根据上下文判断说话人。
# '''
AUDIO_SIM_STR_EN = lambda audio_sim_list_str: f'''
'''

SINGLE_CHAR_RELATION_STR_CN = lambda character1, character2, relationship: f'''
{character1} 是 {character2} 的 {relationship}
'''
SINGLE_CHAR_RELATION_STR_EN = lambda character1, character2, relationship: f'''
{character1} 是 {character2} 的 {relationship}
'''

CHAR_RELATION_STR_CN = lambda char_relation_list_str: f'''
## 查询到的所有人物关系如下：
{char_relation_list_str}
{PROMPT_SEPARATOR}
注：工具给出的人物关系不仅包括目标台词发生时的人物关系，也包括全剧其他部分的人物关系。例如，小明和小李当前是同学也是朋友，但在后续剧情中成为恋人，那么工具会同时给出"同学"、"朋友"、"恋人"这三条关系，请注意甄别和判断。
'''
CHAR_RELATION_STR_EN = lambda char_relation_list_str: f'''
'''

CHAR_RELATION_FAILURE_STR_CN = f'''
## 未能查询到任何人物关系。
'''
CHAR_RELATION_FAILURE_STR_EN = f'''
'''

VIDEO_CAP_BRIEF_STR_CN = lambda video_cap_brief_str: f'''
## 目标台词前后1-2分钟内视频的简要描述如下：
{video_cap_brief_str}
'''
VIDEO_CAP_BRIEF_STR_EN = lambda video_cap_brief_str: f'''
'''

VIDEO_CAP_DETAILED_STR_CN = lambda video_cap_detailed_str: f'''
## 目标台词前后5-10秒内视频的详细描述如下：
{video_cap_detailed_str}
'''
VIDEO_CAP_DETAILED_STR_EN = lambda video_cap_detailed_str: f'''
'''

SYSTEM_PROMPT_CN = f'''
你是一个在影视剧场景下负责台词说话人识别的智能助手。用户会首先给出目标台词的序号和上下文，以及候选说话人名单。请你根据用户提供的信息，结合调用下方介绍的工具获取其余信息，最终判断目标台词对应的说话人。你需要在给出的候选人名单中选出最可能的说话人。
{PROMPT_SEPARATOR}
## 工具介绍：
1. audio_sim - 调用此工具无需提供参数。工具会给出每个真实候选人及场景中可能出现的其他角色的语音库与目标台词语音的声纹相似度。
2. char_relation - 调用此工具无需提供参数。工具会给出每位真实候选人在剧中存在的所有人物关系。
3. video_cap_detailed - 调用此工具无需提供参数。工具会给出目标台词发生时前后5-10秒内的影视剧画面的文字描述，你可以通过这段文字了解到目标台词发生时的画面细节描述。
4. video_cap_brief - 调用此工具无需提供参数。工具会给出目标台词发生前后1-2分钟内的影视剧画面的文字描述，你可以通过这段文字了解到目标台词发生的短期故事背景。
{PROMPT_SEPARATOR}
## 工具调用与结果输出方法：
1. 你总是需要以中文文本格式进行输出，并使用特定标识来分隔输出中各部分的内容。具体而言，你需要使用<think>和</think>作为开头标识和结尾标识来标记你的思考过程。你需要使用<tool>和</tool>作为开头标识和结尾标识来标记你进行的工具调用。你需要使用<answer>和</answer>作为开头标识和结尾标识来标记你给出的最终说话人识别结果。其中：
2. 你的每次回复都必须存在用<think>和</think>标记的思考过程。
3. 你总是需要在"进行工具调用"和"给出最终结果"中二选其一，即：当你还需要调用工具来获取更多信息时，你需要使用<tool>和</tool>标记工具调用信息，此时不能再使用<answer>和</answer>标记结果输出；当你决定给出最终的角色判断时，你需要使用<answer>和</answer>标记结果输出，此时不能再使用<tool>和</tool>标记工具调用。
4. 当你使用<tool>和</tool>标记工具调用信息时，标记内容为你需要调用工具的准确名称。如果调用工具时需要提供参数，你需要按照工具说明中的要求，将参数依次放在工具名称后的英文括号中，并依次用英文逗号分隔。即"工具名称(参数1, 参数2, ...)"。如果不需要提供参数，你只需要在分隔符内输出工具名称。当你进行工具调用后，用户会在下一条消息中提供你所调用工具返回的结果。每次输出只允许调用一个工具。请不要重复调用工具！只要工具名或者参数列表不完全相同，就不视为重复调用；无参数的工具只允许调用一次。
5. 当你使用<answer>和</answer>标记最终结果信息时，标记内容为你给出的最终说话人角色名，即目标台词对应的说话人名称。即"<answer>角色名</answer>"。当你给出了最终结果时，用户会接收你的结果并结束会话。
6. 强调：请不要输出不包含于任何一对标识对范围内的内容。
{PROMPT_SEPARATOR}
## 工具使用的注意事项：
1. 在大部分情况下，声纹相似度是最有效的信息，请确保在推理过程中调用一次"audio_sim"工具。
2. 如果目标台词或与目标台词形成对话关系的上下文台词中有关系称呼词，或目标台词与上下文的表达暗示了情景中角色间的关系，你可以积极尝试调用工具2（即"char_relation"）以得到相关的关系信息。
3. 通常情况下，影视剧场景描述能够为推理提供有价值的线索；除非你根据上述信息能够十分确定说话人，否则请积极调用工具3（即"video_cap_detailed"）获取局部画面描述；如有必要，还可以积极调用工具4（即"video_cap_brief"）获取短期故事背景。
4. ID连续的台词截取自影视剧剧情中的连续片段。然而，给定台词片段不一定对应于某个完整的对话场景，且给定台词不一定恰好构成"video_cap_detailed"所对应的场景，两者对应的影视剧区间可能存在差异。你可以根据台词上下文、"video_cap_detailed"中的画面描述、"video_cap_brief"中的情节描述来推断场景中存在的角色，并根据两种描述信息，结合上下文语境、人物对话关系、人物身份关系、性格、说话风格等线索进行推理，最终判断影视剧中目标台词对应的说话人。
{PROMPT_SEPARATOR}
## 特别强调的注意事项：
1. 请确保你每次输出都包含<think>和</think>标记的思考过程。请确保你每次输出都包含<think>和</think>标记的思考过程。请确保你每次输出都包含<think>和</think>标记的思考过程。
2. 请你充分调用工具后再输出答案。在任何情况下，都应避免第一次回复时直接给出答案。
3. 每次输出只允许调用一个工具。每次输出只允许调用一个工具。每次输出只允许调用一个工具。
4. 请使用中文输出思考过程和答案。
{PROMPT_SEPARATOR}
请你根据当前的已知信息，输出文本格式的回复，其按照上述要求包含所需的标识对和标识对内的内容，不要输出任何多余的信息。
'''
SYSTEM_PROMPT_EN = f'''
'''

USER_PROMPT_CN = USER_PROMPT_EN = lambda subtitle_info_str, candidate_info_str, task_prompt_str: f'''
{subtitle_info_str}
{PROMPT_SEPARATOR}
{candidate_info_str}
{PROMPT_SEPARATOR}
{task_prompt_str}
'''

CALLED_TOOL_INFO_CN = lambda called_tool_info_str: f'''
你已经调用过的工具列表如下：
{called_tool_info_str}
请不要重复调用工具！如果你想要调用的工具都被调用过了，请你从历史对话中查找调用结果，并且在思考后直接给出答案！
''' if len(called_tool_info_str) > 0 else f'''
注意：你还没有调用过任何工具。
'''
CALLED_TOOL_INFO_EN = lambda called_tool_info_str: f'''
'''

class ContentErrorType(Enum):
    PATTERN_NOT_MATCHED = "pattern_not_matched"
    MULTIPLE_IDENTIFIER = "multiple_identifier"
    NO_THINK_IDENTIFIER = "no_think_identifier"
    TOOL_AND_ANSWER = "tool_and_answer"
    UNSUPPORTED_TOOL = "unsupported_tool"
    INVALID_PARAMETER_LIST = "invalid_parameter_list"
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    TOOL_NOT_CALLED_YET = "tool_not_called_yet"
    ANSWER_IS_NOT_CANDIDATE = "answer_is_not_candidate"
ERROR_INFO_FORMAT_CN = "你输出的文本基础格式有误："
ERROR_INFO_CN = {
    ContentErrorType.PATTERN_NOT_MATCHED: "{:s}未按照规定的标识符配对格式输出。".format(ERROR_INFO_FORMAT_CN),
    ContentErrorType.MULTIPLE_IDENTIFIER: lambda key: "{:s}标识符<{:s}>多次出现。".format(ERROR_INFO_FORMAT_CN, key),
    ContentErrorType.NO_THINK_IDENTIFIER: "{:s}标识符<think>未出现。".format(ERROR_INFO_FORMAT_CN),
    ContentErrorType.TOOL_AND_ANSWER: "{:s}未满足标识符<tool>和<answer>出现且仅出现一个的要求。".format(ERROR_INFO_FORMAT_CN),
    ContentErrorType.UNSUPPORTED_TOOL: lambda tool_name: "你给出的工具名{:s}不存在于工具介绍列表中，请确保工具名与工具介绍中存在的工具名完全匹配。".format(tool_name),
    ContentErrorType.INVALID_PARAMETER_LIST: lambda tool_name: "你给出的参数列表和调用工具{:s}所需要的参数数量不匹配，可能你为一个不需要参数的工具传入了参数，或者没有为一个需要参数的工具传入参数。".format(tool_name),
    ContentErrorType.DUPLICATE_TOOL_CALL: lambda tool_func_str: "你重复调用了工具{:s}，请确保工具或者参数与以往的调用不同。".format(tool_func_str),
    ContentErrorType.TOOL_NOT_CALLED_YET: lambda tool_name: "请确保在给出答案前至少调用一次工具{:s}。".format(tool_name),
    ContentErrorType.ANSWER_IS_NOT_CANDIDATE: lambda character_name: '你给出的角色名{:s}不存在于说话人候选名单中，请确保从候选名单中选取答案，其中"其他"表示说话人是除了其余候选人外的其他角色。'.format(character_name),
}
ERROR_INFO_FORMAT_EN = ""
ERROR_INFO_EN = {
    ContentErrorType.PATTERN_NOT_MATCHED: "{:s}".format(ERROR_INFO_FORMAT_EN),
    ContentErrorType.MULTIPLE_IDENTIFIER: lambda key: "{:s}{:s}".format(ERROR_INFO_FORMAT_EN, key),
    ContentErrorType.NO_THINK_IDENTIFIER: "{:s}".format(ERROR_INFO_FORMAT_EN),
    ContentErrorType.TOOL_AND_ANSWER: "{:s}".format(ERROR_INFO_FORMAT_EN),
    ContentErrorType.UNSUPPORTED_TOOL: lambda tool_name: "{:s}".format(tool_name),
    ContentErrorType.INVALID_PARAMETER_LIST: lambda tool_name: "{:s}".format(tool_name),
    ContentErrorType.DUPLICATE_TOOL_CALL: lambda tool_func_str: "{:s}".format(tool_func_str),
    ContentErrorType.TOOL_NOT_CALLED_YET: lambda tool_name: "{:s}".format(tool_name),
    ContentErrorType.ANSWER_IS_NOT_CANDIDATE: lambda character_name: "{:s}".format(character_name),
}
ERROR_PROMPT_CN = lambda error_info: f'''
这是一条报错信息，你上一次的输出将被视为无效。请你根据报错内容重新进行合法的输出，且在输出中不要加入对所犯错误的陈述或反思，直接输出更改后的内容即可。报错信息具体如下：
{PROMPT_SEPARATOR}
{error_info}
'''
ERROR_PROMPT_EN = lambda error_info: f'''
'''

MESSAGE_CN = MESSAGE_EN = lambda role_str, prompt: {
    "role": role_str,
    "content": prompt
}

def get_prompt_func(language, prompt_name):
    return globals()["{:s}_{:s}".format(prompt_name.upper(), language.value.upper())]
