FROM docker.io/pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm
ENV MODEL_DIR=/opt/ml/model
ENV WORKSPACE_DIR=/opt/workspace
ENV PYTHON_VERSION=3.11
ENV INFERENCE_ENV_DIR=/opt/conda/envs/inference-env

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends htop wget unzip vim less git curl && \
    rm -rf /var/lib/apt/lists/*
    
RUN mkdir -p ${MODEL_DIR} && \
    mkdir -p ${WORKSPACE_DIR}
RUN echo "MODEL_DIR: ${MODEL_DIR}"

RUN conda create --yes python=${PYTHON_VERSION} -p ${INFERENCE_ENV_DIR}
ENV PATH=${INFERENCE_ENV_DIR}/bin:$PATH
RUN pip install uv
RUN uv pip list -v

ENV VIRTUAL_ENV=${INFERENCE_ENV_DIR}
ENV CONDA_PREFIX=${INFERENCE_ENV_DIR}

WORKDIR ${WORKSPACE_DIR}
ADD src ${WORKSPACE_DIR}/src
ADD pyproject.toml ${WORKSPACE_DIR}/pyproject.toml
ADD README.md ${WORKSPACE_DIR}/README.md

RUN echo "Installing dependencies..."
RUN uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install -r pyproject.toml --no-build-isolation --extra vllm

ADD entrypoint.sh ${WORKSPACE_DIR}/entrypoint.sh
RUN chmod +x ${WORKSPACE_DIR}/entrypoint.sh
ENTRYPOINT ["/opt/workspace/entrypoint.sh"]






