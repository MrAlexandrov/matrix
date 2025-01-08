PROJECT_NAME = matrix

INPUT_FILE = input.txt
OUTPUT_FILE = output.txt

NPROCS ?= $(shell nproc)

BUILD_DIR = build

all: build test run

build:
	@echo "==> Configuring the project..."
	@cmake -B$(BUILD_DIR) -H.
	@echo "==> Building the project..."
	@cmake --build $(BUILD_DIR) -j $(NPROCS)

test: build
	@echo "==> Running tests..."
	@cd $(BUILD_DIR) && ctest --verbose

run: build
	@echo "==> Running ${PROJECT_NAME}"
	@${BUILD_DIR}/${PROJECT_NAME} < ${INPUT_FILE} > ${OUTPUT_FILE}

clang-tidy:
	clang-tidy include/matrix.hpp main.cpp tests/matrix_test.cpp -- -std=c++20

clean:
	@echo "==> Cleaning up..."
	@rm -rf $(BUILD_DIR)
	@rm -rf coverage matrix.profdata

rebuild: clean build

install:
	sudo apt-get update
	sudo apt-get install -y cmake clang libgtest-dev

coverage: test
	llvm-profdata merge -sparse build/tests/default.profraw -o matrix.profdata
	llvm-cov show ./build/tests/matrix_test -instr-profile=matrix.profdata -format=html -show-branches=count -output-dir=coverage

.PHONY: all build test clang-tidy clean rebuild install
