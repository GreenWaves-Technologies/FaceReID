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
CNN_LAYERS_SRC=$(wildcard quantized_model/*.json)

REID_MODEL_GEN = CnnKernels
REID_MODEL_GEN_C = $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CLEAN = $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

GenReidNet: ../../ReID-Demo/ReIDModel.c ../../ReID-Demo/param_layer_manual.c
	gcc -no-pie -o GenReidNet  -DGRAPH -DNO_CONST -D_FOR_GAPOC_=1 -I"$(TILER_PATH)/include" -I"$(CNN_GEN_PATH)" ../../ReID-Demo/ReIDModel.c ../../ReID-Demo/param_layer_manual.c $(CNN_GEN_PATH)/CNN_Generators.c $(TILER_LIB)

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

APP_SRCS = main.c ../../ReID-Demo/network_process_manual.c ../../ReID-Demo/param_layer_manual.c ../../ReID-Demo/dnn_utils.c ../../ReID-Demo/ImgIO.c $(REID_MODEL_GEN_C) $(CNN_KERNELS_SRC)

APP_CFLAGS += -I$(PWD) -I$(PWD)/../../ReID-Demo -I"$(TILER_INC)" -I"$(CNN_KER_PATH)"

APP_CFLAGS += $(CONTROL_MACRO) -Os

DATA_FILES = ../../ReID-Demo/quantized_model/*.bin

PLPBRIDGE_FLAGS += -f $(DATA_FILES) $(DEVICE_CONNECTION) -fileIO 10

tiler_models: reid_model extra_model

all:: tiler_models

clean::
	rm -rf ./GenReidNet
	rm -rf ./output.bin
	rm -rf $(REID_MODEL_GEN_CLEAN)

.PHONY: reid_model tiler_models all clean

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk
