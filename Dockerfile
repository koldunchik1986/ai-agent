FROM nvidia/cuda:11.8-devel-ubuntu22.04


\u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 \u0441\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0445 \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0435\u0439

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
    && rm -rf /var/lib/apt/lists/*


\u0421\u043e\u0437\u0434\u0430\u0435\u043c \u0432\u0438\u0440\u0442\u0443\u0430\u043b\u044c\u043d\u043e\u0435 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u0435

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.10/site-packages"


\u041a\u043e\u043f\u0438\u0440\u0443\u0435\u043c \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0438 \u0438 \u0443\u0441\u0442\u0430\u043d\u0430\u0432\u043b\u0438\u0432\u0430\u0435\u043c \u0438\u0445

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt


\u0423\u0441\u0442\u0430\u043d\u0430\u0432\u043b\u0438\u0432\u0430\u0435\u043c \u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0435 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f \u0434\u043b\u044f CUDA

ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.1"  # P104-100 architecture
ENV PYTHONUNBUFFERED=1


\u0421\u043e\u0437\u0434\u0430\u0435\u043c \u0440\u0430\u0431\u043e\u0447\u0443\u044e \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u044e

WORKDIR /workspace


\u041a\u043e\u043f\u0438\u0440\u0443\u0435\u043c \u0438\u0441\u0445\u043e\u0434\u043d\u044b\u0439 \u043a\u043e\u0434

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/


\u0421\u043e\u0437\u0434\u0430\u0435\u043c \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0438 \u0434\u043b\u044f \u0434\u0430\u043d\u043d\u044b\u0445 \u0441 \u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u044b\u043c\u0438 \u043f\u0440\u0430\u0432\u0430\u043c\u0438

RUN mkdir -p /home/ai-agent/{models,documents,cache,logs} && \
    chmod 755 /home/ai-agent && \
    chown -R 1000:1000 /home/ai-agent


\u0423\u0441\u0442\u0430\u043d\u0430\u0432\u043b\u0438\u0432\u0430\u0435\u043c \u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0435 \u0434\u043b\u044f \u043f\u0443\u0442\u0435\u0439

ENV AGENT_HOME="/home/ai-agent"
ENV MODEL_CACHE_PATH="/home/ai-agent/models"
ENV DOCUMENT_PATH="/home/ai-agent/documents"
ENV CACHE_PATH="/home/ai-agent/cache"


\u041e\u0442\u043a\u0440\u044b\u0432\u0430\u0435\u043c \u043f\u043e\u0440\u0442 \u0434\u043b\u044f potential web interface

EXPOSE 8000


\u041a\u043e\u043c\u0430\u043d\u0434\u0430 \u0437\u0430\u043f\u0443\u0441\u043a\u0430

CMD ["python", "-m", "src.cli_interface"]