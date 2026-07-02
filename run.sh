# Paths
ROOT_ABSOLUTE_PATH=/opt/huawei/explorer-env/checkpoint/longvideo_data/DramaSR_v2/
DRAMA_DATA_DIR=drama_data/
WORKSPACE_DIR=workspace/

# Drama info
DRAMA_NAME_SFT=ren_shi_jian
DRAMA_NAME_RL=zhen_huan_zhuan
DRAMA_NAME_INFER=chen_mo_de_zhen_xiang,da_qin_di_guo_zhi_zong_heng,huan_le_song,kuang_biao,shan_hai_qing,yi_qi_tong_guo_chuang_1,zhan_chang_sha

# Data arguments
INIT_SEEDS_OPTION=1pct
LABEL_PROP_OPTION=apw
SFT_DATA_OPTION=gpt5c
SFT_WORKERS=64
RL_DATA_OPTION=none
RL_WORKERS=32
# API_KEY=PASTE_YOU_API_KEY_HERE
# BASE_URL=PASTE_YOUR_BASE_URL_HERE
ENFORCE_REFRESH=0
ENFORCE_RETRIAL=1

# Training arguments
NUM_GPUS=8
TARGET_MODEL=q3_8b
SFT_VALIDATION_RATIO=0.1
SFT_ENFORCE_RETRAINING=0
SFT_LEARNING_RATE=0.00001
SFT_NUM_EPOCHS=5
RL_VALIDATION_RATIO=0.1
RL_ENFORCE_RETRAINING=0
RL_LEARNING_RATE=0.000001
RL_NUM_EPOCHS=2
RL_GROUP_SIZE=8
RL_BATCH_SIZE=12
RL_TEMPERATURE=0.6
RL_TOP_P=0.95
RL_PRESENCE_PENALTY=0.1
RL_FREQUENCY_PENALTY=0.3
RL_MAX_TURNS=15
RL_KL_LOSS_COEF=0.0001
RL_KL_COEF=0.05
RL_PENALTY_LENGTH=0.01
RL_PENALTY_DUPLICATE=0.1

# Inference arguments
CONDA_ENV_NAME=speaker_recognition
# TRAINED_MODEL_TIMESTAMP=20260327_190348,20260327_190358,20260327_190408,20260328_031702,20260328_051811,20260328_140954,20260328_153451,20260328_154042,20260329_022958,20260329_023018
TRAINED_MODEL_TIMESTAMP=20260330_193036
INFER_WORKERS=$(expr 32 \* ${NUM_GPUS})
API_KEY_LOCAL=DramaSR
BASE_PORT_ID=8000
INFER_TEMPERATURE=0.6
INFER_TOP_P=0.95

# Switching steps on/off
data_preparation=
data_curation=
model_training=
model_inference=
result_statistics=

if [[ ! -z "$data_preparation" ]]; then
    echo Step 0: Init seeds generation
    python main.py \
        --root_absolute_path ${ROOT_ABSOLUTE_PATH} \
        --drama_data_dir ${DRAMA_DATA_DIR} --workspace_dir ${WORKSPACE_DIR} \
        --drama_name_sft ${DRAMA_NAME_SFT} --drama_name_rl ${DRAMA_NAME_RL} --drama_name_infer ${DRAMA_NAME_INFER} \
        --working_stage data_preparation \
        --init_seeds_option ${INIT_SEEDS_OPTION}
fi

if [[ ! -z "$data_curation" ]]; then
    echo Step 1: Data curation
    python main.py \
        --root_absolute_path ${ROOT_ABSOLUTE_PATH} \
        --drama_data_dir ${DRAMA_DATA_DIR} --workspace_dir ${WORKSPACE_DIR} \
        --drama_name_sft ${DRAMA_NAME_SFT} --drama_name_rl ${DRAMA_NAME_RL} --drama_name_infer ${DRAMA_NAME_INFER} \
        --working_stage data_curation \
        --init_seeds_option ${INIT_SEEDS_OPTION} --label_prop_option ${LABEL_PROP_OPTION} \
        --sft_data_option ${SFT_DATA_OPTION} --sft_workers ${SFT_WORKERS} \
        --rl_data_option ${RL_DATA_OPTION} --rl_workers ${RL_WORKERS} \
        --api_key ${API_KEY} --base_url ${BASE_URL} \
        --enforce_refresh ${ENFORCE_REFRESH} --enforce_retrial ${ENFORCE_RETRIAL}
fi

if [[ ! -z "$model_training" ]]; then
    echo Step 2: Model training
    python main.py \
        --root_absolute_path ${ROOT_ABSOLUTE_PATH} \
        --drama_data_dir ${DRAMA_DATA_DIR} --workspace_dir ${WORKSPACE_DIR} \
        --drama_name_sft ${DRAMA_NAME_SFT} --drama_name_rl ${DRAMA_NAME_RL} --drama_name_infer ${DRAMA_NAME_INFER} \
        --working_stage model_training \
        --init_seeds_option ${INIT_SEEDS_OPTION} --label_prop_option ${LABEL_PROP_OPTION} \
        --sft_data_option ${SFT_DATA_OPTION} --rl_data_option ${RL_DATA_OPTION} \
        --num_gpus ${NUM_GPUS} --target_model ${TARGET_MODEL} \
        --sft_validation_ratio ${SFT_VALIDATION_RATIO} --sft_enforce_retraining ${SFT_ENFORCE_RETRAINING} \
        --sft_learning_rate ${SFT_LEARNING_RATE} --sft_num_epochs ${SFT_NUM_EPOCHS} \
        --rl_validation_ratio ${RL_VALIDATION_RATIO} --rl_enforce_retraining ${RL_ENFORCE_RETRAINING} \
        --rl_learning_rate ${RL_LEARNING_RATE} --rl_num_epochs ${RL_NUM_EPOCHS} \
        --rl_group_size ${RL_GROUP_SIZE} --rl_batch_size ${RL_BATCH_SIZE} \
        --rl_temperature ${RL_TEMPERATURE} --rl_top_p ${RL_TOP_P} \
        --rl_presence_penalty ${RL_PRESENCE_PENALTY} --rl_frequency_penalty ${RL_FREQUENCY_PENALTY} \
        --rl_max_turns ${RL_MAX_TURNS} --rl_kl_loss_coef ${RL_KL_LOSS_COEF} --rl_kl_coef ${RL_KL_COEF} \
        --rl_penalty_length ${RL_PENALTY_LENGTH} --rl_penalty_duplicate ${RL_PENALTY_DUPLICATE}
fi

if [[ ! -z "$model_inference" ]]; then
    echo Step 3: Model inference
    python main.py \
        --root_absolute_path ${ROOT_ABSOLUTE_PATH} \
        --drama_data_dir ${DRAMA_DATA_DIR} --workspace_dir ${WORKSPACE_DIR} \
        --drama_name_sft ${DRAMA_NAME_SFT} --drama_name_rl ${DRAMA_NAME_RL} --drama_name_infer ${DRAMA_NAME_INFER} \
        --working_stage model_inference \
        --init_seeds_option ${INIT_SEEDS_OPTION} --label_prop_option ${LABEL_PROP_OPTION} \
        --num_gpus ${NUM_GPUS} ${CONDA_ENV_NAME:+--conda_env_name "$CONDA_ENV_NAME"} \
        --trained_model_timestamp ${TRAINED_MODEL_TIMESTAMP} \
        --infer_workers ${INFER_WORKERS} --api_key_local ${API_KEY_LOCAL} --base_port_id ${BASE_PORT_ID} \
        --infer_temperature ${INFER_TEMPERATURE} --infer_top_p ${INFER_TOP_P} \
        --enforce_refresh ${ENFORCE_REFRESH} --enforce_retrial ${ENFORCE_RETRIAL}
fi

if [[ ! -z "$result_statistics" ]]; then
    echo Step 4: Result statistics
    python main.py \
        --root_absolute_path ${ROOT_ABSOLUTE_PATH} \
        --drama_data_dir ${DRAMA_DATA_DIR} --workspace_dir ${WORKSPACE_DIR} \
        --drama_name_sft ${DRAMA_NAME_SFT} --drama_name_rl ${DRAMA_NAME_RL} --drama_name_infer ${DRAMA_NAME_INFER} \
        --working_stage result_statistics \
        --init_seeds_option ${INIT_SEEDS_OPTION} --label_prop_option ${LABEL_PROP_OPTION} \
        --trained_model_timestamp ${TRAINED_MODEL_TIMESTAMP} \
        --infer_temperature ${INFER_TEMPERATURE} --infer_top_p ${INFER_TOP_P}
fi
