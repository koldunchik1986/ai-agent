FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    nano \
    && rm -rf /var/lib/apt/lists/*
 
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/workspace"
 
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
 
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV PYTHONUNBUFFERED=1

ENV MAX_JOBS=1
ENV OMP_NUM_THREADS=1
 
WORKDIR /workspace:/workspace/src
 
COPY src/ /workspace
RUN echo " List of /workspace/ after copying:" && \
    ls -la /workspace/ && \
    echo ""

COPY data/ /workspace/data/
RUN echo " List of /workspace/data/ after copying:" && \
    ls -la /workspace/data/ && \
    echo ""

COPY scripts/ /workspace/scripts/
RUN echo " List of /workspace/scripts/ after copying:" && \
    ls -la /workspace/scripts/ && \
    echo ""

RUN mkdir -p /home/ai-agent/{models,documents,cache,logs} && \
    chmod 755 /home/ai-agent && \
    chown -R 1000:1000 /home/ai-agent
 
ENV AGENT_HOME="/home/ai-agent"
ENV MODEL_CACHE_PATH="/home/ai-agent/models"
ENV DOCUMENT_PATH="/home/ai-agent/documents"
ENV CACHE_PATH="/home/ai-agent/cache"
 
EXPOSE 8000
 
CMD ["python", "-m", "cli_interface"]
