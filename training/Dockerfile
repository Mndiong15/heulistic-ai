# Built from the repository root, so that heulistic-axolotl can be installed
# from source:  docker build -f training/Dockerfile .
#
# BASE is overridable so a local check can build against an image already on
# disk instead of pulling the multi-GB Axolotl base.
ARG BASE=axolotlai/axolotl-uv:main-latest
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY training/requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt

# The parallelism planner — the same one the API uses to choose the instance
# and the validator uses to check the config. The trainer calls it to decide
# whether to shard across GPUs.
COPY packages/heulistic-axolotl /tmp/heulistic-axolotl
RUN uv pip install /tmp/heulistic-axolotl && rm -rf /tmp/heulistic-axolotl

# llama.cpp, for export to GGUF: `axolotl export` runs its convert script and
# its llama-quantize binary. Pinned to a release; only the quantizer is built,
# and for CPU — quantizing does not use the GPU.
ARG LLAMA_CPP_REF=b11223
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
 && cmake --build /opt/llama.cpp/build --target llama-quantize -j"$(nproc)" \
 && rm -rf /opt/llama.cpp/.git
ENV LLAMA_CPP_DIR=/opt/llama.cpp

COPY training/train.py training/model_eval.py training/export.py /app/

WORKDIR /app

ENTRYPOINT ["python", "train.py"]
