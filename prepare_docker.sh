#!/bin/bash

# -------------------------------------------------- 以下环境变量需要手动设置 --------------------------------------------------

# 设置模型权重路径环境变量
export AIX4_INSTRUCT_CKPT_PATH="/gxkj/models/Meta-Llama-3.1-8B-Instruct/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
export AIX4_INSTRUCT2_CKPT_PATH="/gxkj/models/Meta-Llama-3.1-70B-Instruct/models--meta-llama--Meta-Llama-3.1-70B-Instruct"
export CONTAINER_NAME="Engine_v5_0"
export DOCKER_IMAGE="vllm/release:240912"

# 端口映射

# HOST模型服务端口
export AIX4_INSTRUCT_MODEL_HOST_PORT="12621"
export AIX4_INSTRUCT2_MODEL_HOST_PORT="12622"


# ------------------------------------------------------ 以下代码请勿改动 ------------------------------------------------------

docker run --name "$CONTAINER_NAME" \
           --detach \
           --gpus all \
           --shm-size=1g \
           --volume "$(pwd)/AixVllm":/aix_model_server \
           --volume $AIX4_INSTRUCT_CKPT_PATH:/weight/ckpt_8b/ \
           --volume $AIX4_INSTRUCT2_CKPT_PATH:/weight/ckpt_76b/ \
           --publish $AIX4_INSTRUCT_MODEL_HOST_PORT:32801 \
           --publish $AIX4_INSTRUCT2_MODEL_HOST_PORT:32802 \
           --workdir /aix_model_server \
           --entrypoint "/bin/bash" \
           $DOCKER_IMAGE -c "tail -f /dev/null"

# ---------------------------------------------------------------------------------------------------------------------------
