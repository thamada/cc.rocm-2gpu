# 2 GPU 間の VRAM 転送と、2 GPU 同時の Host↔VRAM 読み書き。
# 計測内容と帯域の定義は p2p_bw.hip 冒頭のコメントを参照。
#
#   make run
#   make run GPU_A=0 GPU_B=1 MAX_MB=16384 ITERS=0
#
# ITERS=0 はサイズごとに反復回数を自動選択する。

GPU_A  ?= 0
GPU_B  ?= 1
MAX_MB ?= 16384
ITERS  ?= 0

ROCM       ?= /opt/rocm
HIPCC      ?= $(ROCM)/bin/hipcc
ROCMININFO ?= $(ROCM)/bin/rocminfo

GPU_ARCH ?= $(shell $(ROCMININFO) 2>/dev/null | awk '/^  Name:/ { n=$$NF; if (n ~ /^gfx[0-9]+/) { print n; exit } }')

CXX ?= g++
GCC_MACHINE := $(shell $(CXX) -dumpmachine 2>/dev/null)
GCC_VER     := $(shell $(CXX) -dumpversion 2>/dev/null | cut -d. -f1)
GCC_DIR     := /usr/lib/gcc/$(GCC_MACHINE)/$(GCC_VER)
ifneq ($(wildcard $(GCC_DIR)),)
  HIP_GCC_FLAGS := --gcc-install-dir=$(GCC_DIR) -Wno-gcc-install-dir-libstdcxx
endif

TARGET := p2p-bw
SRC    := p2p_bw.hip

HIP_CFLAGS := -O3 -std=c++17 -Wall -Wextra \
	--offload-arch=$(GPU_ARCH) $(HIP_GCC_FLAGS)
HIP_LDFLAGS := -L$(ROCM)/lib -lamdhip64 -Wl,-rpath,$(ROCM)/lib

.PHONY: run build clean detect-gpu-arch
.DEFAULT_GOAL := run

ifeq ($(GPU_ARCH),)
$(error GPU_ARCH を検出できません。$(ROCMININFO) を確認するか、make GPU_ARCH=gfx1201 run)
endif

detect-gpu-arch:
	@echo "GPU_ARCH=$(GPU_ARCH)"

build: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(HIP_CFLAGS) -o $@ $< $(HIP_LDFLAGS)

run: build
	@echo "=== run: GPU_A=$(GPU_A) GPU_B=$(GPU_B) MAX_MB=$(MAX_MB) ITERS=$(ITERS) ==="
	./$(TARGET) $(GPU_A) $(GPU_B) $(MAX_MB) $(ITERS)

clean:
	rm -f $(TARGET)
