USE_PMSIS_BSP=1
DEVICE_CONNECTION ?= -jtag

APP_SRCS = main.c ../../ReID-Demo/ui_input_box.c ../../ReID-Demo/PS2Keyboard.c

APP_CFLAGS += -O3 -g -I../../platforms/gapuino -I../../ReID-Demo/

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk
