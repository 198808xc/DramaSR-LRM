conda create --name speaker_recognition --clone base --offline
conda activate speaker_recognition

pip list | tail -n +3 | awk '{print $1}' | grep -vE "^(pip|setuptools|wheel|distribute)$" | xargs -r pip uninstall -y

python -m pip install --upgrade pip

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 triton==3.2.0 \
    vllm==0.8.5 xformers==0.0.29.post2 outlines-core==0.1.26 \
    opentelemetry-sdk==1.26.0 opentelemetry-semantic-conventions-ai==0.4.13 \
    protobuf==4.25.3 numpy==1.26.4

pip install hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 PyYAML==6.0.2 --no-build-isolation

pip install fire pandas tensordict torchdata codetiming datasets peft libtmux
pip install qwen-vl-utils transformers==4.54.0 tokenizers==0.21.1 huggingface_hub==0.36.0 --force-reinstall --no-deps

pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-deps
