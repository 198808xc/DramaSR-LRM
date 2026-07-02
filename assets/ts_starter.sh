#!/bin/bash

BASE_PORT={{base_port_id}}
GPU_COUNT={{num_gpus}}
MODEL_PATH={{model_path}}
MODEL_NAME={{model_name}}
HOST="127.0.0.1"

declare -a PIDS=()

echo "Starting $GPU_COUNT servers..."
for ((i=0; i<$GPU_COUNT; i++)); do
    PORT=$((BASE_PORT + i))
    echo "Starting server on GPU $i, Port: $PORT"

    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 \
        --host $HOST \
        --port $PORT \
        --served-model-name ${MODEL_NAME}-GPU${i} & \
        --disable-log-requests \
        --disable-log-stats > vllm_server_gpu_${i}.log 2>&1 &

    PIDS[$i]=$!
    sleep 5
    echo "Server is started on GPU $i (PID: ${PIDS[$i]})"
done

echo "----------------------------------------"
echo "All servers have been started!"
echo ""
echo "Summary of status:"
for ((i=0; i<$GPU_COUNT; i++)); do
    PORT=$((BASE_PORT + i))
    echo "GPU $i: Port $PORT, PID ${PIDS[$i]}, Model name ${MODEL_NAME}-GPU${i}"
done
echo "----------------------------------------"
echo "To terminate all servers, please execute: kill ${PIDS[*]}"

wait
