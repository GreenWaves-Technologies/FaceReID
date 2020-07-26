# Copyright 2019-2020 GreenWaves Technologies, SAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

USE_PMSIS_BSP := 1

TEST_LAYER_INDEX ?= 0

REID_PATH := ../../ReID-Demo
CNN_LAYERS_PATH := $(REID_PATH)/quantized_model
CNN_LAYERS_SRC := $(wildcard $(CNN_LAYERS_PATH)/*.json)
CNN_LAYERS_BIN := $(CNN_LAYERS_SRC:.json=.weights.bin) $(CNN_LAYERS_SRC:.json=.bias.bin)

%.bias.bin %.weights.bin: %.json
	../../scripts/convert2gap.py $< $(basename $<)

REID_MODEL_GEN := CnnKernels
REID_MODEL_GEN_C := $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CH := $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

REID_GEN_SRCS := \
  $(REID_PATH)/ReIDModel.c \
  $(REID_PATH)/FireGenerator.c \
  $(REID_PATH)/layer_params.c \
  $(TILER_CNN_GENERATOR_PATH)/CNN_Generators.c \
  $(TILER_CNN_GENERATOR_PATH)/CNN_Generator_Util.c
REID_GEN_CFLAGS += -fcommon -I"$(TILER_INC)" -I"$(TILER_CNN_GENERATOR_PATH)"

GenReidNet: $(REID_GEN_SRCS)
	gcc -o $@ $(REID_GEN_CFLAGS) $(REID_GEN_SRCS) $(TILER_LIB)

$(REID_MODEL_GEN_C): GenReidNet
	./GenReidNet

reid_model: $(REID_MODEL_GEN_C) $(CNN_LAYERS_BIN)

main.c: $(REID_MODEL_GEN_C)

APP_SRCS := \
  main.c \
  $(REID_PATH)/layer_params.c \
  $(REID_PATH)/dnn_utils.c \
  $(REID_MODEL_GEN_C) \
  $(TILER_CNN_KERNEL_PATH)/CNN_Conv_BasicKernels.c \
  $(TILER_CNN_KERNEL_PATH)/CNN_BiasReLULinear_BasicKernels.c \
  $(TILER_CNN_KERNEL_PATH)/CNN_Pooling_BasicKernels.c
APP_INC += $(PWD) $(REID_PATH) "$(TILER_INC)" "$(TILER_CNN_KERNEL_PATH)"
APP_CFLAGS += -O3 -DNORM_ROUND -DTEST_LAYER_INDEX=$(TEST_LAYER_INDEX)
ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
endif

READFS_FILES += $(CNN_LAYERS_BIN)

tiler_models: reid_model

build: tiler_models

clean::
	rm -f GenReidNet
	rm -f output.bin
	rm -f $(REID_MODEL_GEN_CH)

.PHONY: reid_model tiler_models build clean

include $(RULES_DIR)/pmsis_rules.mk
