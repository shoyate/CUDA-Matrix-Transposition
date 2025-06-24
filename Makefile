# CUDA Matrix Transposition Makefile
# Note: On Windows, you need Visual Studio environment set up
# Run from Visual Studio Developer Command Prompt or use build_and_run.bat

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_50 -std=c++11

# Target executable (Windows)
TARGET = matrix_transpose.exe

# Source files
SOURCES = main.cu

# Default target
all: $(TARGET)

# Compile CUDA program
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

# Clean target
clean:
	del /f $(TARGET) matrix_transpose.exe 2>nul

# Run the program
run: $(TARGET)
	$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the matrix transpose program"
	@echo "  clean   - Remove executable files"
	@echo "  run     - Build and run the program"
	@echo "  help    - Show this help message"

.PHONY: all clean run help
