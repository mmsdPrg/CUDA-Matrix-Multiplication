#!/bin/bash
# HW4 CUDA - WSL Setup Script

set -e

echo "=========================================="
echo "HW4 CUDA Matrix Multiplication"
echo "Setup for WSL"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_error() {
    echo -e "${RED}[!]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running in WSL
if ! grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    print_error "This script must be run in WSL!"
    exit 1
fi

print_success "Running in WSL"

# Step 1: Check System Requirements
echo ""
echo "=========================================="
echo "📋 Step 1: Checking System Requirements"
echo "=========================================="

# Check for GPU
print_status "Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
    print_success "GPU detected!"
else
    print_error "nvidia-smi not found! Please install NVIDIA drivers in Windows."
    exit 1
fi

# Check for basic tools
print_status "Checking build tools..."
if ! command -v gcc &> /dev/null; then
    print_warning "gcc not found, installing build-essential..."
    sudo apt update
    sudo apt install -y build-essential
fi
print_success "Build tools ready"

# Check for Python
print_status "Checking Python..."
if ! command -v python3 &> /dev/null; then
    print_warning "Python3 not found, installing..."
    sudo apt install -y python3 python3-pip
fi
PYTHON_VERSION=$(python3 --version)
print_success "$PYTHON_VERSION ready"

# Step 2: Install/Check CUDA
echo ""
echo "=========================================="
echo "🔧 Step 2: Setting up CUDA Toolkit"
echo "=========================================="

if ! command -v nvcc &> /dev/null; then
    print_warning "CUDA Toolkit not found, installing..."
    
    # Download and install CUDA keyring
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    
    # Install CUDA toolkit
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-11-8
    
    # Setup PATH
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    
    # Add to bashrc
    if ! grep -q "CUDA PATH" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CUDA PATH" >> ~/.bashrc
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
else
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

nvcc --version
print_success "CUDA Toolkit ready"

# Step 3: Install Python Dependencies
echo ""
echo "=========================================="
echo "🐍 Step 3: Installing Python Dependencies"
echo "=========================================="

print_status "Installing Python packages..."
pip3 install --upgrade pip --break-system-packages -q

# Install PyTorch with CUDA support
if ! python3 -c "import torch" 2>/dev/null; then
    print_status "Installing PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --break-system-packages -q
fi

# Install other requirements
pip3 install numpy matplotlib pandas --break-system-packages -q 2>/dev/null || true

# Verify installations
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"

print_success "Python dependencies installed"

# Step 4: Detect GPU Architecture
echo ""
echo "=========================================="
echo "🎯 Step 4: Detecting GPU Architecture"
echo "=========================================="

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

print_status "GPU: $GPU_NAME"
print_status "Compute Capability: $COMPUTE_CAP"

# Convert compute capability to SM architecture
if [[ "$COMPUTE_CAP" == "7.5" ]]; then
    GPU_ARCH="sm_75"
    print_status "Architecture: Turing (sm_75)"
elif [[ "$COMPUTE_CAP" == "8.0" ]]; then
    GPU_ARCH="sm_80"
    print_status "Architecture: Ampere (sm_80)"
elif [[ "$COMPUTE_CAP" == "8.6" ]]; then
    GPU_ARCH="sm_86"
    print_status "Architecture: Ampere (sm_86)"
elif [[ "$COMPUTE_CAP" == "8.9" ]]; then
    GPU_ARCH="sm_89"
    print_status "Architecture: Ada Lovelace (sm_89)"
else
    GPU_ARCH="sm_75"
    print_warning "Unknown compute capability, using sm_75"
fi

# Update Makefile with correct architecture
if [ -f "Makefile" ]; then
    print_status "Updating Makefile with GPU architecture..."
    sed -i "s/GPU_ARCH = sm_[0-9]*/GPU_ARCH = $GPU_ARCH/" Makefile
    print_success "Makefile updated with $GPU_ARCH"
fi

# Step 5: Create Results Directory
echo ""
echo "=========================================="
echo "📁 Step 5: Setting Up Project Structure"
echo "=========================================="

mkdir -p results/logs results/profiles
print_success "Directories created"

# Step 6: Quick Compilation Test
echo ""
echo "=========================================="
echo "🔨 Step 6: Testing CUDA Compilation"
echo "=========================================="

print_status "Compiling baseline kernel..."
nvcc -arch=$GPU_ARCH -O3 -Xcompiler -fPIC --shared matmul_kernel.cu -o libmatmul.so

if [ -f "libmatmul.so" ]; then
    print_success "Compilation successful!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Step 7: Quick Runtime Test
echo ""
echo "=========================================="
echo "🧪 Step 7: Running Quick Test"
echo "=========================================="

print_status "Running quick test (1 epoch)..."
python3 train.py --use-custom --epochs 1 2>&1 | tee test_output.txt

if grep -q "Accuracy=" test_output.txt; then
    ACCURACY=$(grep "Accuracy=" test_output.txt | tail -1 | sed -n 's/.*Accuracy=\([0-9.]*\)%.*/\1/p')
    if (( $(echo "$ACCURACY > 50" | bc -l) )); then
        print_success "Test passed! Accuracy: ${ACCURACY}%"
    else
        print_warning "Low accuracy: ${ACCURACY}% (normal for 1 epoch)"
    fi
    rm test_output.txt
else
    print_error "Test failed!"
    cat test_output.txt
    exit 1
fi

# Step 8: Ready to Run Full Experiments
echo ""
echo "=========================================="
echo "🎉 Setup Complete! Ready to Run"
echo "=========================================="
echo ""
echo "Your system is ready! You can now:"
echo ""
echo -e "${GREEN}Quick commands:${NC}"
echo "  make help              # Show all available commands"
echo "  make baseline          # Run baseline experiments"
echo "  make run-all           # Run ALL experiments (30-45 min)"
echo ""
echo -e "${GREEN}Step by step:${NC}"
echo "  make memory            # Memory allocation tests"
echo "  make blocksize         # Block size optimization"
echo "  make precision         # Precision experiments"
echo "  make unroll            # Loop unrolling tests"
echo "  make profile           # Profiling (requires nsys)"
echo ""
echo -e "${GREEN}Analysis:${NC}"
echo "  python3 analyze_results.py    # Generate reports and graphs"
echo "  python3 analyze_profiles.py   # Analyze profiling data"
echo ""
echo -e "${YELLOW}GPU Configuration:${NC}"
echo "  GPU: $GPU_NAME"
echo "  Architecture: $GPU_ARCH"
echo "  Compute Capability: $COMPUTE_CAP"
echo ""
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Ready to start! Run: ${YELLOW}make run-all${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""
