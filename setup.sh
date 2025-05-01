#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Progress animation function
progress() {
    local duration=$1
    local message=$2
    local width=30
    local progress=0
    local bar_char="▇"
    local empty_char="░"
    
    echo -ne "\n${CYAN}${message}${NC}\n"
    
    while [ $progress -le $width ]; do
        local bar=""
        for ((i=0; i<$progress; i++)); do
            bar="${bar}${bar_char}"
        done
        for ((i=$progress; i<$width; i++)); do
            bar="${bar}${empty_char}"
        done
        
        percentage=$((progress * 100 / width))
        echo -ne "\r[${GREEN}${bar}${NC}] ${percentage}%"
        progress=$((progress + 1))
        sleep $(echo "scale=3; $duration/$width" | bc)
    done
    echo -e "\n"
}

# Download YOLO11 models function
download_models() {
    local models_dir="models"
    local base_url="https://github.com/ultralytics/assets/releases/download/v8.3.0"
    declare -A model_sizes=(
        ["n"]="2.6M"
        ["s"]="9.4M"
        ["m"]="20.1M"
        ["l"]="25.3M"
        ["x"]="56.9M"
    )
    
    echo -e "\n${YELLOW}Downloading YOLO11 model weights...${NC}"
    
    # Create models directory if it doesn't exist
    mkdir -p "$models_dir"
    
    # Download each model size
    for size in n s m l x; do
        local model_name="yolo11${size}.pt"
        local url="${base_url}/${model_name}"
        local output="${models_dir}/${model_name}"
        
        if [ -f "$output" ]; then
            echo -e "${BLUE}ℹ ${model_name} (${model_sizes[$size]}) already exists${NC}"
            continue
        fi
        
        echo -e "\n${CYAN}Downloading ${model_name} (${model_sizes[$size]})...${NC}"
        if curl -# -L "$url" -o "$output"; then
            echo -e "${GREEN}✓ Successfully downloaded ${model_name}${NC}"
        else
            echo -e "${RED}✗ Failed to download ${model_name}${NC}"
            return 1
        fi
    done
    
    echo -e "\n${GREEN}✓ All base models downloaded successfully${NC}"
    return 0
}

# Print header
echo -e "\n${BOLD}${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║     YOLO11 Live Demo - Setup Assistant      ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════╝${NC}\n"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version)
    echo -e "${GREEN}✓ Found ${python_version}${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8 or later.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Setting up Python virtual environment...${NC}"
if [ -d "yolo_env" ]; then
    echo -e "${BLUE}ℹ Found existing environment, removing...${NC}"
    rm -rf yolo_env
fi

python3 -m venv yolo_env
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Virtual environment created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create virtual environment${NC}"
    exit 1
fi

progress 1 "Activating virtual environment..."
source yolo_env/bin/activate

# Install requirements
echo -e "${YELLOW}Installing required packages...${NC}"
echo -e "${CYAN}This may take a few minutes depending on your internet connection.${NC}\n"

pip install --upgrade pip >/dev/null 2>&1
progress 2 "Upgrading pip..."

pip install -r requirements.txt 2>&1 | while read -r line; do
    if [[ $line == *"Successfully installed"* ]]; then
        echo -e "${GREEN}✓${NC} $line"
    elif [[ $line == *"ERROR"* ]]; then
        echo -e "${RED}✗${NC} $line"
    fi
done

# Create required directories and download models
echo -e "\n${YELLOW}Setting up project directories...${NC}"

# Create models directory
if [ ! -d "models" ]; then
    mkdir models
    echo -e "${GREEN}✓ Created models directory${NC}"
else
    echo -e "${BLUE}ℹ Models directory already exists${NC}"
fi

# Download YOLO11 models
download_models

# Create Streamlit config directory and file
echo -e "\n${YELLOW}Configuring Streamlit...${NC}"
mkdir -p .streamlit
cat > .streamlit/config.toml << EOL
[server]
fileWatcherType = "none"
runOnSave = false

[browser]
gatherUsageStats = false

[global]
developmentMode = false
EOL
echo -e "${GREEN}✓ Created Streamlit configuration${NC}"

# Final message
echo -e "\n${BOLD}${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║             Setup Complete! 🎉              ║${NC}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════╝${NC}\n"

echo -e "${CYAN}To start the demo:${NC}"
echo -e "1. ${YELLOW}Activate the environment:${NC}"
echo -e "   ${GREEN}source yolo_env/bin/activate${NC}"
echo -e "2. ${YELLOW}Run the demo:${NC}"
echo -e "   ${GREEN}streamlit run aio.py${NC}\n"

echo -e "${BLUE}Happy detecting! 🚀${NC}\n" 