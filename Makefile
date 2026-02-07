# HW4 CUDA Matrix Multiplication Optimization Makefile
# Author: [Your Name]
# Date: February 2026

# Compiler and flags
NVCC = nvcc
PYTHON = python3

# CUDA architecture (adjust based on your GPU)
# sm_75 = Turing (T4, RTX 20 series)
# sm_86 = Ampere (RTX 30 series)
# sm_89 = Ada Lovelace (RTX 40 series)
GPU_ARCH = sm_75

# Compiler flags:
# -arch: Specify GPU architecture
# -O3: Maximum optimization level
# -Xcompiler -fPIC: Generate position-independent code for shared library
# --shared: Create a shared library (.so)
NVCC_FLAGS = -arch=$(GPU_ARCH) -O3 -Xcompiler -fPIC --shared

# Source files
SOURCES = matmul_kernel.cu matmul_managed.cu \
          matmul_block8.cu matmul_block16.cu matmul_block32.cu \
          matmul_fp16.cu matmul_int8.cu \
          matmul_unroll.cu matmul_pragma.cu

# Output shared libraries
LIBRARIES = $(SOURCES:.cu=.so)

# Training parameters
EPOCHS = 2
BATCH_SIZE = 64

# Directory for results
RESULTS_DIR = results
PROFILE_DIR = $(RESULTS_DIR)/profiles
LOG_DIR = $(RESULTS_DIR)/logs

.PHONY: all clean baseline memory blocksize precision unroll profile run-all help setup

# Default target
all: setup $(LIBRARIES)

# Setup directories
setup:
	@echo "Creating result directories..."
	@mkdir -p $(RESULTS_DIR) $(PROFILE_DIR) $(LOG_DIR)

# Help target
help:
	@echo "HW4 CUDA Matrix Multiplication Makefile"
	@echo "========================================"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Compile all CUDA kernels"
	@echo "  baseline    - Run baseline implementation"
	@echo "  memory      - Run memory allocation experiments"
	@echo "  blocksize   - Run block size experiments"
	@echo "  precision   - Run precision experiments"
	@echo "  unroll      - Run loop unrolling experiments"
	@echo "  profile     - Run profiling with nsys and ncu"
	@echo "  run-all     - Run all experiments"
	@echo "  clean       - Remove all compiled files and results"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Compiler Flags:"
	@echo "  -arch=$(GPU_ARCH): GPU architecture"
	@echo "  -O3: Maximum optimization"
	@echo "  -Xcompiler -fPIC: Position-independent code"
	@echo "  --shared: Create shared library"

# Generic rule to compile .cu to .so
%.so: %.cu
	@echo "Compiling $< to $@..."
	$(NVCC) $(NVCC_FLAGS) $< -o lib$@

# Specific compilation rules with proper naming
libmatmul.so: matmul_kernel.cu
	@echo "Compiling baseline kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@

libmatmul_managed.so: matmul_managed.cu
	@echo "Compiling managed memory kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

libmatmul_block8.so: matmul_block8.cu
	@echo "Compiling 8x8 block kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

libmatmul_block16.so: matmul_block16.cu
	@echo "Compiling 16x16 block kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

libmatmul_block32.so: matmul_block32.cu
	@echo "Compiling 32x32 block kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

libmatmul_fp16.so: matmul_fp16.cu
	@echo "Compiling FP16 kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

libmatmul_int8.so: matmul_int8.cu
	@echo "Compiling INT8 kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

libmatmul_unroll.so: matmul_unroll.cu
	@echo "Compiling unrolled kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

libmatmul_pragma.so: matmul_pragma.cu
	@echo "Compiling pragma unroll kernel..."
	$(NVCC) $(NVCC_FLAGS) $< -o libmatmul.so

# Step 1: Baseline
baseline: libmatmul.so
	@echo ""
	@echo "===== Running Baseline Experiments ====="
	@echo "PyTorch baseline..."
	$(PYTHON) train.py --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/pytorch_baseline.log
	@echo ""
	@echo "CUDA baseline..."
	$(NVCC) $(NVCC_FLAGS) matmul_kernel.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/cuda_baseline.log

# Step 2: Memory allocation strategies
memory: setup
	@echo ""
	@echo "===== Running Memory Allocation Experiments ====="
	@echo "Standard cudaMalloc..."
	$(NVCC) $(NVCC_FLAGS) matmul_kernel.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/memory_cudamalloc.log
	@echo ""
	@echo "cudaMallocManaged..."
	$(NVCC) $(NVCC_FLAGS) matmul_managed.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/memory_managed.log

# Step 3: Block size optimization
blocksize: setup
	@echo ""
	@echo "===== Running Block Size Experiments ====="
	@echo "Block size 8x8..."
	$(NVCC) $(NVCC_FLAGS) matmul_block8.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/blocksize_8x8.log
	@echo ""
	@echo "Block size 16x16..."
	$(NVCC) $(NVCC_FLAGS) matmul_block16.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/blocksize_16x16.log
	@echo ""
	@echo "Block size 32x32..."
	$(NVCC) $(NVCC_FLAGS) matmul_block32.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/blocksize_32x32.log

# Step 4: Precision experiments
precision: setup
	@echo ""
	@echo "===== Running Precision Experiments ====="
	@echo "FP32 baseline..."
	$(NVCC) $(NVCC_FLAGS) matmul_kernel.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/precision_fp32.log
	@echo ""
	@echo "FP16..."
	$(NVCC) $(NVCC_FLAGS) matmul_fp16.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/precision_fp16.log
	@echo ""
	@echo "INT8..."
	$(NVCC) $(NVCC_FLAGS) matmul_int8.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/precision_int8.log

# Step 5: Loop unrolling
unroll: setup
	@echo ""
	@echo "===== Running Loop Unrolling Experiments ====="
	@echo "Manual unrolling..."
	$(NVCC) $(NVCC_FLAGS) matmul_unroll.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/unroll_manual.log
	@echo ""
	@echo "Pragma unrolling..."
	$(NVCC) $(NVCC_FLAGS) matmul_pragma.cu -o libmatmul.so
	$(PYTHON) train.py --use-custom --epochs $(EPOCHS) 2>&1 | tee $(LOG_DIR)/unroll_pragma.log

# Step 6: Profiling
profile: setup
	@echo ""
	@echo "===== Running Profiling with nsys ====="
	$(NVCC) $(NVCC_FLAGS) matmul_kernel.cu -o libmatmul.so
	nsys profile --stats=true -o $(PROFILE_DIR)/baseline python train.py --use-custom --epochs 1
	nsys stats $(PROFILE_DIR)/baseline.nsys-rep > $(PROFILE_DIR)/baseline_stats.txt
	@echo ""
	@echo "Profiling best configuration..."
	$(NVCC) $(NVCC_FLAGS) matmul_block16.cu -o libmatmul.so
	nsys profile --stats=true -o $(PROFILE_DIR)/optimized python train.py --use-custom --epochs 1
	nsys stats $(PROFILE_DIR)/optimized.nsys-rep > $(PROFILE_DIR)/optimized_stats.txt

# Run all experiments
run-all: setup baseline memory blocksize precision unroll
	@echo ""
	@echo "===== All Experiments Completed ====="
	@echo "Results saved in $(RESULTS_DIR)"
	@echo "Generating summary..."
	$(PYTHON) analyze_results.py

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f *.so *.o
	rm -rf $(RESULTS_DIR)
	rm -rf data
	rm -rf __pycache__
	rm -f *.nsys-rep *.ncu-rep

# Check GPU info
gpu-info:
	@echo "GPU Information:"
	@nvidia-smi
	@echo ""
	@echo "CUDA Version:"
	@nvcc --version
