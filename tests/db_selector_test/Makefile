USE_PMSIS_BSP=1
DEVICE_CONNECTION ?= -jtag

APP_SRCS = main.c ../../ReID-Demo/face_db.c ../../ReID-Demo/dnn_utils.c

APP_CFLAGS += -I$(PWD) -I$(PWD)/../../ReID-Demo  -I"$(TILER_PATH)/include" -I"$(CNN_KER_PATH)/include"

FACE_DB_SIZE=$(shell wc -l < ./known_faces/index.txt)
DATA_FILES = ./known_faces/index.txt ./known_faces/*.bin

APP_CFLAGS += -DSTATIC_FACE_DB=1 -DFACE_DB_SIZE=$(FACE_DB_SIZE)

PLPBRIDGE_FLAGS += -f $(DATA_FILES) $(DEVICE_CONNECTION) -fileIO 10

.PHONY: all clean

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk
