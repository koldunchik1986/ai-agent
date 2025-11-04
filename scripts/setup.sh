
#!/bin/bash

# \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043e\u0447\u043d\u044b\u0439 \u0441\u043a\u0440\u0438\u043f\u0442 \u0434\u043b\u044f AI-\u0430\u0433\u0435\u043d\u0442\u0430 \u043d\u0430 \u0431\u0430\u0437\u0435 Mistral AI 7B
# \u041f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0430 Kali Linux, CUDA 11.8, GPU P104-100

set -e

echo "\ud83d\ude80 \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 AI-\u0430\u0433\u0435\u043d\u0442\u0430 \u043d\u0430 \u0431\u0430\u0437\u0435 Mistral AI 7B"
echo "============================================"

# \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u043f\u0440\u0430\u0432 root
if [[ $EUID -ne 0 ]]; then
   echo "\u274c \u042d\u0442\u043e\u0442 \u0441\u043a\u0440\u0438\u043f\u0442 \u0434\u043e\u043b\u0436\u0435\u043d \u0431\u044b\u0442\u044c \u0437\u0430\u043f\u0443\u0449\u0435\u043d \u0441 \u043f\u0440\u0430\u0432\u0430\u043c\u0438 root"
   echo "   \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439\u0442\u0435: sudo ./setup.sh"
   exit 1
fi

# \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u041e\u0421
if ! grep -q "Kali" /etc/os-release; then
    echo "\u26a0\ufe0f  \u041f\u0440\u0435\u0434\u0443\u043f\u0440\u0435\u0436\u0434\u0435\u043d\u0438\u0435: \u0421\u0438\u0441\u0442\u0435\u043c\u0430 \u043d\u0435 \u044f\u0432\u043b\u044f\u0435\u0442\u0441\u044f Kali Linux"
    read -p "\u041f\u0440\u043e\u0434\u043e\u043b\u0436\u0438\u0442\u044c \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0443? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 CUDA
echo "\ud83d\udd0d \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "\u274c NVIDIA \u0434\u0440\u0430\u0439\u0432\u0435\u0440 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d. \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u0435 NVIDIA \u0434\u0440\u0430\u0439\u0432\u0435\u0440\u044b:"
    echo "   sudo apt update"
    echo "   sudo apt install nvidia-driver-470"
    echo "   sudo reboot"
    exit 1
fi

# \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0432\u0435\u0440\u0441\u0438\u0438 CUDA
CUDA_VERSION=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | grep -o '[0-9]*\.[0-9]*')
echo "\u2705 \u041d\u0430\u0439\u0434\u0435\u043d\u0430 CUDA \u0432\u0435\u0440\u0441\u0438\u044f: $CUDA_VERSION"

if [[ $(echo "$CUDA_VERSION < 11.8" | bc -l) -eq 1 ]]; then
    echo "\u26a0\ufe0f  \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f CUDA 11.8 \u0438\u043b\u0438 \u0432\u044b\u0448\u0435"
fi

# \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0439
echo "\ud83d\udcc1 \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0439..."
BASE_DIR="/home/sda3/ai-agent"
mkdir -p $BASE_DIR/{models,documents,cache,logs,neo4j/{data,logs,import},chroma}
chown -R $SUDO_USER:$SUDO_USER $BASE_DIR
echo "\u2705 \u0414\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0438 \u0441\u043e\u0437\u0434\u0430\u043d\u044b \u0432 $BASE_DIR"

# \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 Docker
echo "\ud83d\udc33 \u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 Docker..."
if ! command -v docker &> /dev/null; then
    echo "\ud83d\udce6 \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 Docker..."
    apt-get update
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # \u0414\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044f \u0432 \u0433\u0440\u0443\u043f\u043f\u0443 docker
    usermod -aG docker $SUDO_USER
    echo "\u2705 Docker \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d. \u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c $SUDO_USER \u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d \u0432 \u0433\u0440\u0443\u043f\u043f\u0443 docker"
else
    echo "\u2705 Docker \u0443\u0436\u0435 \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d"
fi

# \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "\ud83d\udce6 \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "\u2705 Docker Compose \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d"
else
    echo "\u2705 Docker Compose \u0443\u0436\u0435 \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d"
fi

# \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 Docker GPU Support
echo "\ud83c\udfae \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 GPU \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0438..."
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "\ud83d\udce6 \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    
    apt-get update
    apt-get install -y nvidia-docker2
    systemctl restart docker
    echo "\u2705 NVIDIA Container Toolkit \u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d"
else
    echo "\u2705 GPU \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0430 \u0443\u0436\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d\u0430"
fi

# \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0444\u0430\u0439\u043b\u0430 \u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0445 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f
echo "\ud83d\udd27 \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0445 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f..."
cat > /etc/environment.d/ai-agent.conf << EOF
AGENT_HOME="/home/sda3/ai-agent"
MODEL_CACHE_PATH="/home/sda3/ai-agent/models"
DOCUMENT_PATH="/home/sda3/ai-agent/documents"
CACHE_PATH="/home/sda3/ai-agent/cache"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password"
CHROMA_HOST="localhost"
CHROMA_PORT="8001"
CUDA_VISIBLE_DEVICES="0"
EOF

echo "\u2705 \u041f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u044b\u0435 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f \u0441\u043e\u0437\u0434\u0430\u043d\u044b"

# \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 \u0441\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0445 \u043b\u0438\u043c\u0438\u0442\u043e\u0432
echo "\u2699\ufe0f  \u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0430 \u0441\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0445 \u043b\u0438\u043c\u0438\u0442\u043e\u0432..."
cat > /etc/security/limits.d/ai-agent.conf << EOF
$SUDO_USER soft nofile 65536
$SUDO_USER hard nofile 65536
$SUDO_USER soft nproc 32768
$SUDO_USER hard nproc 32768
EOF

echo "\u2705 \u0421\u0438\u0441\u0442\u0435\u043c\u043d\u044b\u0435 \u043b\u0438\u043c\u0438\u0442\u044b \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043d\u044b"

# \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u043d\u043e\u0433\u043e \u0444\u0430\u0439\u043b\u0430
echo "\ud83d\udd27 \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 systemd \u0441\u0435\u0440\u0432\u0438\u0441\u0430..."
cat > /etc/systemd/system/ai-agent.service << EOF
[Unit]
Description=AI Agent Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PWD
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-agent
echo "\u2705 systemd \u0441\u0435\u0440\u0432\u0438\u0441 \u0441\u043e\u0437\u0434\u0430\u043d"

# \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 Python \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0435\u0439 \u0434\u043b\u044f \u0445\u043e\u0441\u0442\u0430 (\u0434\u043b\u044f \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u043a\u0438)
echo "\ud83d\udc0d \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 Python \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0435\u0439..."
if ! command -v python3 &> /dev/null; then
    apt-get install -y python3 python3-pip python3-venv
fi

# \u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0432\u0438\u0440\u0442\u0443\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u044f \u0434\u043b\u044f \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u043a\u0438
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "\u2705 \u0412\u0438\u0440\u0442\u0443\u0430\u043b\u044c\u043d\u043e\u0435 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u0435 \u0441\u043e\u0437\u0434\u0430\u043d\u043e"
else
    echo "\u2705 \u0412\u0438\u0440\u0442\u0443\u0430\u043b\u044c\u043d\u043e\u0435 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u0435 \u0443\u0436\u0435 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u0435\u0442"
fi

echo ""
echo "\ud83c\udf89 \u0423\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u0430!"
echo "======================"
echo ""
echo "\ud83d\udccb \u0421\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0435 \u0448\u0430\u0433\u0438:"
echo "1. \u041f\u0435\u0440\u0435\u0437\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 \u0441\u0438\u0441\u0442\u0435\u043c\u0443 \u0438\u043b\u0438 \u0432\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u0435:"
echo "   source /etc/environment.d/ai-agent.conf"
echo ""
echo "2. \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 \u0441\u0435\u0440\u0432\u0438\u0441\u044b:"
echo "   sudo systemctl start ai-agent"
echo "   \u0438\u043b\u0438"
echo "   docker-compose up -d"
echo ""
echo "3. \u041f\u0440\u043e\u0432\u0435\u0440\u044c\u0442\u0435 \u0441\u0442\u0430\u0442\u0443\u0441:"
echo "   docker-compose ps"
echo ""
echo "4. \u0417\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 CLI \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441:"
echo "   docker-compose exec ai-agent python -m src.cli_interface"
echo ""
echo "5. \u0414\u043b\u044f \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u043a\u0438 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439\u0442\u0435 \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u043e\u0435 \u043e\u043a\u0440\u0443\u0436\u0435\u043d\u0438\u0435:"
echo "   source venv/bin/activate"
echo "   python -m src.cli_interface"
echo ""
echo "\ud83d\udcc1 \u0414\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u0438:"
echo "   \u041c\u043e\u0434\u0435\u043b\u0438: $BASE_DIR/models"
echo "   \u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b: $BASE_DIR/documents"
echo "   \u041a\u044d\u0448: $BASE_DIR/cache"
echo ""
echo "\ud83c\udf10 Web \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\u044b:"
echo "   Neo4j: http://localhost:7474 (neo4j/password)"
echo "   Chroma: http://localhost:8001"
echo ""
echo "\ud83d\udcd6 \u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u044f: docs/README.md"
