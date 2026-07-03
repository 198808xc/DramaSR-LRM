# DramaSR-LRM

Official code release for the ICML 2026 paper **"Reasoning LLM Improves Speaker Recognition in Long-form TV Dramas"**.

This repository provides a complete pipeline for speaker recognition in long-form TV dramas: pseudo-label generation via label propagation, reasoning-trajectory curation, supervised fine-tuning (SFT), reinforcement learning (RL) with tool use, and evaluation on held-out dramas.

## Overview

Speaker recognition in long TV dramas is challenging because characters appear sparsely, voices change, and off-screen speech is common. DramaSR-LRM addresses this by training a **reasoning LLM** that iteratively calls tools (audio similarity, character relations, video captions, etc.) before producing a final answer.

The pipeline consists of five stages:

| Stage | Script flag in `run.sh` | Description |
|-------|-------------------------|-------------|
| 0. Data preparation | `data_preparation=1` | Generate initial seeds and run label propagation |
| 1. Data curation | `data_curation=1` | Build SFT trajectories and RL datapoints via LLM API calls |
| 2. Model training | `model_training=1` | SFT with LLaMA-Factory, then RL (GRPO) with verl-tool |
| 3. Model inference | `model_inference=1` | Launch vLLM servers and run tool-augmented inference |
| 4. Result statistics | `result_statistics=1` | Aggregate metrics and run error analysis |

## Requirements

- **OS**: Ubuntu 24.04 (tested on GPU servers; the provided scripts assume bash)
- **GPU**: 8× NVIDIA GPUs recommended (`NUM_GPUS=8` in `run.sh`)
- **Python**: 3.10
- **Conda**: required for environment setup
- **Model**: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) placed under `assets/Qwen3-8B/`
- **API access** (for data curation only): a proprietary LLM endpoint (e.g., GPT-5) via `API_KEY` and `BASE_URL` in `run.sh`

Training outputs are written to `/cache/DramaSR_output/` (see `utils.py`). Create this directory or adjust the path before running.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/DramaSR-LRM.git
cd DramaSR-LRM
```

### 2. Set up the GPU environment

Run the provided installation script, which creates a conda environment named `speaker_recognition` and installs PyTorch, vLLM, transformers, and related dependencies:

```bash
bash install.sh
```

Before running, download the FlashAttention wheel referenced in `install.sh`:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Place the `.whl` file in the repository root, then run `install.sh`.

### 3. Install bundled training frameworks

After activating the conda environment:

```bash
conda activate speaker_recognition

# LLaMA-Factory (SFT)
pip install -e LLaMA-Factory/

# verl-tool (RL with tool use; includes a modified verl submodule)
pip install -e verl-tool/
pip install -e verl-tool/verl/
```

### 4. Download the base model

Download **Qwen3-8B** from Hugging Face and place the checkpoint under:

```
assets/Qwen3-8B/
```

Example using `huggingface-cli`:

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir assets/Qwen3-8B
```

## Data

Each drama lives under `drama_data/<drama_name>/` and contains:

| Subdirectory / file | Description |
|---------------------|-------------|
| `subtitle_data/` | Per-episode subtitle JSON with utterance text and **speaker recognition ground truth (GT)** labels |
| `face_data/` | Per-episode face detection/recognition results |
| `sv_embedding/` | Speaker-verification embeddings (`.npy` per episode) |
| `caption.json` | Video scene captions |
| `relation.json` | Character relationship graph |
| `init_seeds_*.json` | Initial labeled utterances (generated in Step 0) |
| `prop_labels_*.json` | Propagated pseudo-labels (generated in Step 0) |

> **Note:** Under `ren_shi_jian/` and `zhen_huan_zhuan/`, there are zip archives that must be **unzipped manually** before the data can be used for training.
>
> **Special note:** `ren_shi_jian/sft_data_1pct_apw_gpt5c.json` is the GPT-5-generated SFT training data for *A Lifelong Journey*.

### Supported dramas

| Drama name | Directory name | Episodes | Language |
|------------|----------------|----------|----------|
| The Long Night | `chen_mo_de_zhen_xiang` | 12 | Chinese |
| Qin Empire 2 | `da_qin_di_guo_zhi_zong_heng` | 51 | Chinese |
| Ode to Joy 1 | `huan_le_song` | 42 | Chinese |
| The Knockout | `kuang_biao` | 39 | Chinese |
| A Lifelong Journey | `ren_shi_jian` | 58 | Chinese |
| Minning Town | `shan_hai_qing` | 23 | Chinese |
| Standing by Me 1 | `yi_qi_tong_guo_chuang_1` | 34 | Chinese |
| Battle of Changsha | `zhan_chang_sha` | 32 | Chinese |
| Empresses in the Palace | `zhen_huan_zhuan` | 76 | Chinese |

Additional dramas (1 Chinese and 3 English) will be released by **July 20, 2026**.

## Quick Start

### Configure paths and dramas

Edit **`run.sh`** before running:

1. Set `ROOT_ABSOLUTE_PATH` to the absolute path of this repository.
2. Choose dramas for SFT, RL, and inference via `DRAMA_NAME_SFT`, `DRAMA_NAME_RL`, and `DRAMA_NAME_INFER` (comma-separated).
3. For data curation, uncomment and set `API_KEY` and `BASE_URL`.
4. Enable the pipeline stages you want by setting the corresponding variable to `1`:

```bash
data_preparation=1
data_curation=1
model_training=1
model_inference=1
result_statistics=1
```

### Run the pipeline

```bash
bash run.sh
```

Each stage invokes **`main.py`** with `--working_stage` set to one of: `data_preparation`, `data_curation`, `model_training`, `model_inference`, or `result_statistics`.

You can also call stages individually:

```bash
python main.py \
  --root_absolute_path /path/to/DramaSR-LRM \
  --drama_data_dir drama_data/ \
  --workspace_dir workspace/ \
  --drama_name_sft ren_shi_jian \
  --drama_name_rl zhen_huan_zhuan \
  --drama_name_infer chen_mo_de_zhen_xiang \
  --working_stage data_preparation \
  --init_seeds_option 1pct
```

### Reasoning tools

During SFT, RL, and inference, the model may call these tools (defined in `large_reasoning_model/toolset.py`):

- `audio_sim` — speaker-verification similarity against candidates
- `char_relation` — character relationship lookup
- `video_cap_brief` — short-term scene context
- `video_cap_detailed` — detailed visual description

The model responds with structured **think**, **tool**, and **answer** blocks (e.g., `<tool>audio_sim()</tool>`, `<answer>角色名</answer>`).

## File Structure

Key files are **bolded**.

```
DramaSR-LRM/
├── **main.py**                          # Pipeline entry point; orchestrates all five stages
├── **run.sh**                           # Master launcher — configure paths, dramas, and stage flags here
├── **install.sh**                       # GPU conda environment setup (PyTorch, vLLM, FlashAttention, etc.)
├── **utils.py**                         # Global paths, drama metadata, filename conventions, template helpers
├── **data_loader.py**                   # Loads subtitle, face, embedding, caption, and relation data per drama
├── LICENSE
│
├── **assets/**                          # Model checkpoints and training/inference templates
│   ├── **Qwen3-8B/**                    # ← Download Qwen3-8B here (not included in repo)
│   ├── **sft_config.yaml**              # LLaMA-Factory SFT config template (filled at runtime)
│   ├── **rl_starter.sh**                # verl-tool GRPO training launcher template
│   └── **ts_starter.sh**                # vLLM multi-GPU inference server launcher template
│
├── **drama_data/**                      # Per-drama datasets (subtitles, faces, embeddings, captions, …)
│   └── <drama_name>/
│       ├── subtitle_data/               # Per-episode subtitle JSON (0001.json, 0002.json, …)
│       ├── face_data/                   # Per-episode face detection JSON
│       ├── sv_embedding/                # Speaker-verification embeddings (.npy)
│       ├── caption.json
│       ├── relation.json
│       ├── init_seeds_1pct.json         # Generated by label propagation (Step 0)
│       └── prop_labels_1pct_apw.json    # Generated by label propagation (Step 0)
│
├── **label_propagation/**               # Step 0: pseudo-label generation
│   ├── **label_propagation.py**         # Init-seed selection and label propagation orchestration
│   ├── label_structure.py               # Label set data structure
│   ├── graph_algorithm.py               # Graph-based similarity and clustering algorithms
│   ├── c_functions.c / c_caller.py      # C-accelerated graph routines
│   └── c_functions.so
│
├── **large_reasoning_model/**           # Steps 1–4: data curation, training, inference, evaluation
│   ├── **data_curation.py**             # SFT trajectory and RL datapoint generation
│   ├── **model_training.py**            # Wraps LLaMA-Factory (SFT) and verl-tool (RL)
│   ├── **model_inference.py**           # vLLM server management and inference orchestration
│   ├── **toolset.py**                   # Tool definitions and environment interaction logic
│   ├── **verl_toolset.py**              # Offline toolset adapter for verl-tool server
│   ├── **llm_prompts.py**               # Chinese/English prompt templates
│   └── api_caller.py                    # API client for proprietary LLM data curation
│
├── **LLaMA-Factory/**                   # Bundled SFT framework (modified; see Acknowledgements)
│   ├── src/llamafactory/                # Core training code
│   └── examples/deepspeed/ds_z3_config.json  # DeepSpeed ZeRO-3 config used for full fine-tuning
│
├── **verl-tool/**                       # Bundled RL + tool-use framework (modified; see Acknowledgements)
│   ├── verl_tool/
│   │   ├── servers/tools/
│   │   │   └── **speaker_recognition.py**   # Custom tool server for drama speaker recognition
│   │   └── trainer/                     # GRPO / PPO training configs
│   ├── verl/
│   │   └── verl/utils/reward_score/
│   │       └── **__init__.py**          # Custom `speaker_recognition` reward function
│   └── examples/train/speaker_recognition.sh  # Reference RL training script
│
└── workspace/                           # Versioned experiment outputs (SFT/RL data, configs, meta.json)
    └── <version_identifier>_<timestamp>/
        ├── meta.json
        ├── sft_data_all_*.json
        ├── rl_data_all_*.parquet
        └── dataset_info.json
```

### Output directories (created at runtime)

| Path | Contents |
|------|----------|
| `workspace/<version>_<timestamp>/` | Curated SFT/RL datasets and experiment metadata |
| `/cache/DramaSR_output/<version>_<timestamp>/` | SFT and RL model checkpoints |
| `cached_models/` (under repo root) | Cached trained models for inference |

## Acknowledgements

This project builds on two open-source frameworks, included as subdirectories with project-specific modifications:

### [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

Used for full-parameter supervised fine-tuning of Qwen3-8B. We use the bundled `qwen3_nothink` chat template and DeepSpeed ZeRO-3 configuration.

### [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) / [verl](https://github.com/volcengine/verl)

Used for multi-turn GRPO training with tool calling. Project-specific changes include:

- `verl-tool/verl_tool/servers/tools/speaker_recognition.py` — drama-aware tool server
- `verl-tool/verl/verl/utils/reward_score/__init__.py` — rule-based speaker recognition reward
- `assets/rl_starter.sh` — GRPO training recipe for this task

Please cite the original works if you use these components.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026reasoning,
  title     = {Reasoning LLM Improves Speaker Recognition in Long-form TV Dramas},
  author    = {Yuxuan Li and Lingxi Xie and Xinyue Huo and Jihao Qiu and Pengfei Chen and Jiannan Ge and Kaiwen Duan and Qi Tian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

*(Update author list and BibTeX key when the camera-ready version is available.)*

## License

This repository is released under the [MIT License](LICENSE).

Bundled third-party code (LLaMA-Factory, verl-tool, verl) retains its original licenses — see the respective subdirectories.

## Release Plan

| Date | Milestone |
|------|-----------|
| **July 4, 2026** | Pre-trained models (SFT and RL checkpoints) |
| **July 20, 2026** | Remaining dramas (1 Chinese + 3 English) |
