# Copyright 2019 GreenWaves Technologies, SAS
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

STATIC_FACE_DB=1
USE_PMSIS_BSP=1
ifeq ($(BOARD_NAME),gapoc_a)
  DEVICE_CONNECTION = -ftdi
else
  DEVICE_CONNECTION = -jtag
endif

TILER_PATH = $(GAP_SDK_HOME)/tools/autotiler_v3
TILER_LIB = $(TILER_PATH)/lib/libtile.a
TILER_GENERATOR_PATH = $(TILER_PATH)/generators
CNN_GEN_PATH = $(TILER_GENERATOR_PATH)/CNN
CNN_KER_PATH = $(TILER_GENERATOR_PATH)/CNN
CNN_KERNELS_SRC = $(wildcard $(CNN_KER_PATH)/CNN_*.c)

REID_MODEL_GEN = CnnKernels
REID_MODEL_GEN_C = $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CLEAN = $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

GenReidNet: ../../ReID-Demo/ReIDModel.c ../../ReID-Demo/param_layer_manual.c
	gcc -no-pie -o GenReidNet -DGRAPH -DNO_CONST -DCONFIG_GAPOC_A -I"$(TILER_INC)" -I"$(CNN_GEN_PATH)" ../../ReID-Demo/ReIDModel.c ../../ReID-Demo/param_layer_manual.c $(CNN_GEN_PATH)/CNN_Generators.c $(TILER_LIB)

$(REID_MODEL_GEN_C): GenReidNet
	./GenReidNet

reid_model: $(REID_MODEL_GEN_C) $(CNN_LAYERS_BIN)

EXTRA_MODEL_GEN = ExtraKernels
EXTRA_MODEL_GEN_C = $(addsuffix .c, $(EXTRA_MODEL_GEN))
EXTRA_MODEL_GEN_CLEAN = $(EXTRA_MODEL_GEN_C) $(addsuffix .h, $(EXTRA_MODEL_GEN))

GenExtraTile: ../../ReID-Demo/ExtraModels.c
	gcc -no-pie -o GenExtraTile -I"$(TILER_INC)" -I"$(CNN_GEN_PATH)" ../../ReID-Demo/ExtraModels.c $(TILER_LIB)

$(EXTRA_MODEL_GEN_C): GenExtraTile
	./GenExtraTile

extra_model: $(EXTRA_MODEL_GEN_C)

tiler_models: reid_model extra_model

APP_SRCS = main.c ../../ReID-Demo/reid_pipeline.c ../../ReID-Demo/network_process_manual.c ../../ReID-Demo/param_layer_manual.c ../../ReID-Demo/dnn_utils.c ../../ReID-Demo/ExtraBasicKernels.c ../../ReID-Demo/ImgIO.c $(REID_MODEL_GEN_C) $(CNN_KERNELS_SRC) $(EXTRA_MODEL_GEN_C)

APP_CFLAGS += -I$(PWD) -I$(PWD)/../../ReID-Demo -I"$(TILER_INC)" -I"$(CNN_KER_PATH)"
APP_CFLAGS += -Os -DTEST_RESPONSE=test_response_l$(TEST_LEVEL)

PLPBRIDGE_FLAGS += -f $(DATA_FILES) $(DEVICE_CONNECTION) -fileIO

all:: tiler_models

clean::
	rm -rf GenReidNet
	rm -rf GenExtraTile
	rm -rf $(REID_MODEL_GEN_CLEAN) $(EXTRA_MODEL_GEN_CLEAN) $(DETECTION_MODEL_GEN_CLEAN)

.PHONY: reid_model extra_model tiler_models all clean

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk
