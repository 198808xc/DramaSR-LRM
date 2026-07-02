import os
import sys
import time
import json
import re
from enum import Enum
import httpx
import openai
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from toolset import Toolset


class APICaller:

    MODEL_NAME_DICT = {**MODEL_NAME_DICT_PROPRIETARY, **MODEL_NAME_DICT_LOCAL}
    LOCAL_TIMEOUT = 10

    MAX_RETRIES_PROPRIETARY = 3
    MAX_RETRIES_LOCAL = 2
    SLEEP_DURATION_QPM = 30
    class APIFailureType(Enum):
        MAX_RETRIES_REACHED = "max_retries_reached"

    def __init__(self, model_proprietary=None, model_local=None, api_key=None, base_url=None, base_port_id=None, num_gpus=None, args=None):
        assert (model_proprietary is None) != (model_local is None), "Error: exactly one of model_proprietary and model_local shall be provided."
        self.model_proprietary, self.model_local = model_proprietary, model_local
        self.model_name = None
        self.api_key = api_key
        if self.model_proprietary is not None:
            assert model_proprietary in APICaller.MODEL_NAME_DICT, "Error: an unknown proprietary model was specified."
            assert base_url is not None, "Error: base_url for calling the proprietary model shall be provided."
            self.model_proprietary = APICaller.MODEL_NAME_DICT[model_proprietary]
            self.base_url = base_url
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.model_name = self.model_proprietary
            self.api_parameters = {
                "max_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.1,
                "response_format": {"type": "text"}
            }
        else:
            assert model_local in APICaller.MODEL_NAME_DICT, "Error: an unknown local model was specified."
            self.model_local = APICaller.MODEL_NAME_DICT[model_local]
            assert 0 <= base_port_id, "Error: base_port_id must be a non-negative integer."
            assert 0 < num_gpus, "Error: num_gpus must be a positive integer."
            assert args is not None, "Error: args must be present to pass RL parameters."
            self.base_port_id, self.num_gpus = base_port_id, num_gpus
            self.client_dict = {}
            for gpu_id in range(self.num_gpus):
                port_id = self.base_port_id + gpu_id
                self.client_dict.update({
                    gpu_id: {
                        "client": OpenAI(
                            api_key=self.api_key,
                            base_url=LOCAL_PORT_URL(port_id),
                            timeout=APICaller.LOCAL_TIMEOUT
                        ),
                        "model_name": "{:s}-GPU{:d}".format(self.model_local, gpu_id),
                    }
                })
            self.api_parameters = {
                "max_tokens": 1024,
                "temperature": args.infer_temperature,
                "top_p": args.infer_top_p,
                "presence_penalty": args.rl_presence_penalty,
                "frequency_penalty": args.rl_frequency_penalty,
                "stream": False
            }
            print(self.api_parameters)

    def get_info(self):
        return {
            "model_name": self.model_name,
            "api_parameters": self.api_parameters,
        }

    def call_api_proprietary(self, message_list, retries=None, verbose=False):
        assert self.model_proprietary is not None, "Error: the proprietary model was not specified!"
        real_retries = retries if retries is not None else APICaller.MAX_RETRIES_PROPRIETARY
        for _ in range(real_retries):
            try:
                response = self.client.chat.completions.create(model=self.model_proprietary, messages=message_list, **self.api_parameters)
                return True, response.choices[0].message.content
            except openai.RateLimitError as err:
                if verbose:
                    print_with_indent(">> QPM limit was reached; will sleep for {:d} seconds.".format(APICaller.SLEEP_DURATION_QPM), indent=INDENT_LOG)
                time.sleep(APICaller.SLEEP_DURATION_QPM)
            except openai.OpenAIError as err:
                if verbose:
                    print_with_indent(">> An error was returned: {:s} | {:s}.".format(str(type(err)), str(err)), indent=INDENT_LOG)
        print_with_indent(">> The API call failed after {:d} retries; the caller was terminated.".format(real_retries), indent=INDENT_LOG)
        return False, APICaller.APIFailureType.MAX_RETRIES_REACHED

    def call_api_local(self, message_list, gpu_id, retries=None, verbose=False):
        assert self.model_local is not None, "Error: the local model was not specified!"
        real_retries = retries if retries is not None else APICaller.MAX_RETRIES_LOCAL
        for _ in range(real_retries):
            client_info = self.client_dict[gpu_id]
            client, model_name = client_info["client"], client_info["model_name"]
            try:
                response = client.chat.completions.create(model=model_name, messages=message_list, **self.api_parameters)
                if self.api_parameters.get("stream", False):
                    collected_messages = []
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            collected_messages.append(chunk.choices[0].delta.content)
                    content = "".join(collected_messages)
                else:
                    content = response.choices[0].message.content
                return True, content
            except Exception as err:
                if True:
                    print_with_indent(">> An error was returned: {:s}".format(str(err)), indent=INDENT_LOG)
        print_with_indent(">> The API call failed after {:d} retries; the caller was terminated.".format(real_retries), indent=INDENT_LOG)
        return False, APICaller.APIFailureType.MAX_RETRIES_REACHED

    def call_api(self, message_list, gpu_id=None, verbose=False):
        if self.model_proprietary is not None:
            return self.call_api_proprietary(message_list, verbose=verbose)
        else:
            assert gpu_id is not None, "Error: GPU ID must be specified for local API calls."
            assert gpu_id in self.client_dict, "Error: an unknown GPU ID was specified."
            return self.call_api_local(message_list, gpu_id, verbose=verbose)
