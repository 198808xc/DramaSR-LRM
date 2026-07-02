import os
import sys
import subprocess
import time
import json
import select
import libtmux

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *


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
                if unit["identifier"] == "tool":
                    tool_name = unit["text"]
                    if tool_name in called_tool_set:
                        duplicate_calls += 1
                    else:
                        called_tool_set.add(tool_name)
                elif unit["identifier"] == "answer":
                    answer_list.append(unit["text"])
            rounds += 1
    if len(answer_list) != 1:
        return -1
    else:
        answer = answer_list[0]
        reward = (answer == ground_truth["character_gt"]) or \
            (not answer in ground_truth["candidate_list"] and \
             ground_truth["character_gt"] == ground_truth["name_others"])
        return reward - 0.1 * max(rounds - 3, 0) - 0.5 * duplicate_calls


class ModelInference:

    IDLE_THRESHOLD = 60
    TIMEOUT_THRESHOLD = 120

    def __init__(self, project):
        self.project = project
        self.server = None
        self.session = None

    def _get_default_pane(self):
        return self.session.windows[0].panes[0]

    def start_server___(self, args, script_file, session_name="default"):
        print_with_indent("Starting the server...", indent=INDENT_INFO)
        self.server = libtmux.Server()
        if self.server.has_session(session_name):
            self.server.kill_session(session_name)
        self.session = self.server.new_session(session_name=session_name)
        pane = self._get_default_pane()
        if args.conda_env_name is not None:
            pane.send_keys(CONDA_ENV_COMMAND(args.conda_env_name))
            time.sleep(0.1)
        pane.send_keys("bash {:s}".format(script_file))
        start_time = time.time()
        last_output, last_change_time = [], start_time
        print_with_indent("Waiting for server to initialize...", indent=INDENT_LOG)
        started_servers = set()
        while len(started_servers) < args.num_gpus:
            current_output, current_time = pane.capture_pane(), time.time()
            if current_output != last_output:
                last_output, last_change_time = current_output, current_time
                last_output_ = "" if last_output == [] else last_output[-1]
                print_with_indent("Latest output: {:s}".format(last_output_), indent=INDENT_LOG)
            else:
                if (current_time - last_change_time >= ModelInference.IDLE_THRESHOLD) or \
                    (current_time - start_time >= ModelInference.TIMEOUT_THRESHOLD):
                    print_with_indent("Timeout!", indent=INDENT_LOG)
                    self.session.kill_session()
                    return
            for server_idx in range(args.num_gpus):
                if server_idx in started_servers:
                    continue
                url = LOCAL_PORT_URL(args.base_port_id + server_idx)
                try:
                    response = urllib.request.urlopen(url, timeout=ModelInference.VLLM_TIMEOUT)
                    print("!!!!!!", url, response)
                    if response.getcode() == 200:
                        print_with_indent("vLLM server on port {:d} is started!".format(server_idx), indent=INDENT_LOG)
                        started_servers.add(server_idx)
                except (URLError, HTTPError, ConnectionResetError) as e:
                    print("!!!!!!", url, e)
                    pass
            time.sleep(1)
        print_with_indent("Server successfully started.", indent=INDENT_INFO)

    def start_server(self, args, script_file, session_name="default"):
        print_with_indent("Starting the server...", indent=INDENT_INFO)
        self.server = libtmux.Server()
        if self.server.has_session(session_name):
            self.server.kill_session(session_name)
        self.session = self.server.new_session(session_name=session_name)
        pane = self._get_default_pane()
        if args.conda_env_name is not None:
            pane.send_keys(CONDA_ENV_COMMAND(args.conda_env_name))
            time.sleep(0.1)
        pane.send_keys("bash {:s}".format(script_file))
        last_output, last_change_time, has_produced_output = [], time.time(), False
        print_with_indent("Waiting for server to initialize...", indent=INDENT_LOG)
        while True:
            current_output, current_time = pane.capture_pane(), time.time()
            if current_output != last_output:
                last_output, last_change_time, has_produced_output = current_output, current_time, True
                print_with_indent("Latest output: {:s}".format(str(last_output)), indent=INDENT_LOG)
            else:
                time_since_last_change = current_time - last_change_time
                print_with_indent("No input for {:0.1f} seconds.".format(time_since_last_change)) 
                if has_produced_output and time_since_last_change >= ModelInference.IDLE_THRESHOLD:
                    print_with_indent("Server is considered fully started.", indent=INDENT_LOG)
                    break
                if not has_produced_output and time_since_last_change >= ModelInference.TIMEOUT_THRESHOLD:
                    print_with_indent("Timeout waiting for the server to produce initial output.", indent=INDENT_LOG)
                    self.session.kill_session()
                    return
            time.sleep(1)
        print_with_indent("Server successfully started.", indent=INDENT_INFO)

    def stop_server(self):
        print_with_indent("Stopping the server...", indent=INDENT_INFO)
        pane = self._get_default_pane()
        # for _ in range(10):
        #     pane.send_keys('C-c', enter=False)
        #     time.sleep(0.1)
        # time.sleep(5)
        self.session.kill_session()
        self.server = self.session = None
        print_with_indent("Server successfully stopped.", indent=INDENT_INFO)


    # def start_server(self, script_file):
    #     print_with_indent("Starting the server...", indent=INDENT_INFO)
    #     self.server_process = subprocess.Popen(["bash", script_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #     fd = self.server_process.stdout.fileno()
    #     os.set_blocking(fd, False)
    #     last_output_time = time.time()
    #     has_started_outputting = False
    #     while True:
    #         if self.server_process.poll() is not None:
    #             print_with_indent("Server process terminated unexpectedly.", indent=INDENT_INFO)
    #             return
    #         ready, _, _ = select.select([fd], [], [], 0.5)
    #         if ready:
    #             data = os.read(fd, 4096)
    #             if data:
    #                 last_output_time = time.time()
    #                 has_started_outputting = True
    #         else:
    #             current_time = time.time()
    #             if has_started_outputting and (current_time - last_output_time >= 10):
    #                 print_with_indent("No output for 10 seconds. Server is considered fully started.", indent=INDENT_LOG)
    #                 break
    #             elif not has_started_outputting and (current_time - last_output_time >= 60):
    #                 print_with_indent("Timeout waiting for the server to produce initial output.", indent=INDENT_LOG)
    #                 return
    #     print_with_indent("Server successfully started.", indent=INDENT_INFO)

    # def stop_server(self):
    #     print_with_indent("Stopping the server...", indent=INDENT_INFO)
    #     self.server_process.terminate()
    #     try:
    #         self.server_process.wait(timeout=5)
    #     except subprocess.TimeoutExpired:
    #         self.server_process.kill()
    #     print_with_indent("Server successfully stopped.", indent=INDENT_INFO)

    def perform_inference(self, script_file):
        ...
