# Overview

Gapoc A board includes Bluetooth LE module that makes possible smart door bell management with Android-based smartphone remotely. During regular re-id loop thew demo stores all strangers descriptor and photo in L2 and L3 memory accordingly. On button press the board is switched to strangers management mode and activates Bluetooth module for smart phone connection. Strangers management consists of three phases: strangers information download to phone, users management on phone and approved users upload back to device memory. Strangers download to phone and further upload to the device is covered in dedicated sections bellow. At the end of strangers administration step all strangers are dropped from L2 and L3 memory as all known people has been already added to list of users.

# Strangers Download to Android App

![](images/ble_protocol_read.png)

# Known Users Database Update

![](images/ble_protocol_write.png)

# Administration Mode Test

The demo repository includes simple administration mode test in `tests/ble_test` folder. The test application for Gapoc A board loads photos and descriptors for Lena and Francesco from host to strangers list and invokes full users administration pipeline on push button press. Please do the following steps:

0. Install and run GAP ReID app on you Android Phone.
1. Go to test app folder and run test with the command:
```
    $ cd tests/ble_test
    $ make run
```
2. Wait for message `Waiting for button press event` in console output. It means that test application is ready to handle events.
3. Press push button on board and wait for `AT Config Done` message in console.
4. Press "SCAN" button in Android app and wait for Bluetooth device enumeration.
5. Tap on "GreenWaves-GAPOC" device in list and wait for the device connection. Please press "Connect" button in the device screen if device is opened in "Disconnected" state.
6. Press "Refresh" in Android app and wait for photos download. You should get 2 people in strangers list: male and female.
7. Tap on users and input name for one of them and press drop for another.
8. Press push button on the board to exit from users administration mode. The test dumps all new introduced to host with user name in file name.
9. Compare dumped descriptor with original one with `cmp` command to ensure that it's transferred correctly:
```
    $ cmp -l ./Francesco.bin <dumped file>.bin # for Francesco
    $ cmp -l ./Lena.bin <dumped file>.bin # for Lena
```

# Known Issues and Hardware Limitations

1. Due to hardware bug in GAP8 chip Hyperram and UART cannot be used in the same time. Demo downloads strangers photos to L2 memory and switch `HYPERBUS_DATA6_PAD` mode to UART mode before Bluetooth module enabling and back after the disconnection.

2. GAP8 chip does not support hardware flow control and interacts with Ublocks NINA BLE module using DMA ignoring it's state. Maximum DMA transfer size is 1 KiB. In case if sent or received buffer is larger than maximum transfer size several DMA calls are done without BLE module await that leads to data corruption. The demo splits face photo buffer on 1Kib chunks to work around the issue.

3. UART transactions are not buffered and should not be mixed with high latency operations, including `printf` call for logging to console.
