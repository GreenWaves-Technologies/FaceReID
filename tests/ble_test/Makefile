APP = ble_test

USE_PMSIS_BSP=1
BOARD_NAME = gapoc_a

APP_SRCS += ble_main.c ../../ReID-Demo/BleUserManager.c ../../ReID-Demo/dnn_utils.c ../../ReID-Demo/strangers_db.c ../../ReID-Demo/face_db.c ../../ReID-Demo/ImgIO.c $(PLATFORM_SRCS)

APP_CFLAGS += -DUSE_PMSIS=1 -DUSE_BLE_USER_MANAGEMENT=1 -I$(PWD) -I$(PWD)/../../ReID-Demo

PLPBRIDGE_FLAGS += -ftdi -fileIO 10

include $(RULES_DIR)/pmsis_rules.mk
