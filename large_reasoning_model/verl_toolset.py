import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DRAMA_INFO, OFFSCREEN_KEYWORD_LIST, PROP_LABELS_FILE
from data_loader import DramaData
from label_propagation.label_structure import LabelSet
from large_reasoning_model.toolset import Toolset


class MiniArguments:
    def __init__(self):
        pass

def get_offline_toolset(project_dir, drama_data_dir, drama_name, label_prop_str):
    args = MiniArguments()
    setattr(args, "drama_name", drama_name)
    setattr(args, "language", DRAMA_INFO[drama_name]["language"])
    setattr(args, "episodes", DRAMA_INFO[drama_name]["episodes"])
    setattr(args, "drama_dir", os.path.join(project_dir, drama_data_dir, drama_name))
    drama_data = DramaData(None, args).load_all_drama_data()
    label_set = LabelSet(drama_data)
    initial_character_info = drama_data.get_character_info()
    label_set.reset(list(initial_character_info.keys()), OFFSCREEN_KEYWORD_LIST(args.language))
    prop_labels_file = os.path.join(project_dir, drama_data_dir, drama_name, PROP_LABELS_FILE(label_prop_str))
    label_set.load_from(prop_labels_file)
    return Toolset(args.language, drama_data, label_set)
