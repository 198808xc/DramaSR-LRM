import os
import sys
import logging

from .base import BaseTool, register_tool
project_dir = os.environ.get("project_dir")
drama_data_dir = os.environ.get("drama_data_dir")
version_dir = os.environ.get("version_dir")
drama_name_list = os.environ.get("drama_name_list").split(",")
label_prop_option_list = os.environ.get("label_prop_option_list").split(",")
sys.path.append(project_dir)
from large_reasoning_model.verl_toolset import get_offline_toolset

logger = logging.getLogger(__name__)


@register_tool
class SpeakerRecognitionTool(BaseTool):

    tool_type = "speaker_recognition"

    def __init__(self, num_workers=1):
        super().__init__(num_workers=num_workers)
        self.toolset_dict = {}
        assert len(drama_name_list) == len(label_prop_option_list), "Error: the list lengths do not match."
        for drama_name, label_prop_option in zip(drama_name_list, label_prop_option_list):
            self.toolset_dict.update({drama_name: get_offline_toolset(
                project_dir, drama_data_dir, drama_name, label_prop_option)})
        self.trajectory_dict = {}

    def load_env(self, trajectory_id):
        env = self.env_cache.get(trajectory_id)
        if env == None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                    "tool_info_list": [],
                },
                "previous_obs": [],
            }
        return env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        env["metadata"]["turns"] += 1
        if is_valid and action is not None:
            env["metadata"]["tool_info_list"].append(action)
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def conduct_action(self, trajectory_id, action, extra_field):
        env = self.load_env(trajectory_id)
        drama_name, subtitle_idx, sample_type = extra_field["drama_name"], extra_field["subtitle_idx"], extra_field["sample_type"]
        toolset = self.toolset_dict[drama_name]
        message_info = {
            "cheating": False,
            "tool_info_list": env["metadata"]["tool_info_list"],
        }
        done, is_valid, tool_info, observation = toolset.get_next_user_prompt(subtitle_idx, action, message_info, in_rl_training=True)
        if trajectory_id not in self.trajectory_dict:
            self.trajectory_dict.update({trajectory_id: len(self.trajectory_dict)})
        self.update_env(trajectory_id, env, tool_info, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, is_valid
