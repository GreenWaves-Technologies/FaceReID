# Intro

The project is aimed to create and deploy a smart doorbell application that would be highly energy-efficient. The main application pipeline consists of 4 stages: frame capture, face detection, face recognition and user interaction e.g. displaying message. In addition to regular face detection and re-identification loop, application provides functionality to add known people to the trusted list using already calculated face descriptors.

![](documentation/images/reid_pipeline.png)

# Target Platforms

The demo is targeted for two boards Gapoc A v2 and Gapuino v2 and GAP SDK release 3.5 and 3.6. GAP8 V1 chips are not supported in SDK any more.
List of extra components is provided below.

For Gapuino board:
- HIGHMAX camera module
- Adafruit 2.8 TFT display with SPI interface

For Gapoc A board:
- Adafruit 2.8 TFT display with SPI interface
- GAPOC_A Adapter for Adafruit LCD 2.8 version 3 or version 4 or hand-made shield with push button
- Android-based smartphone with [pre-built user management application](https://reid-artifacts.s3.eu-central-1.amazonaws.com/FaceID/ReID-Control-App.apk)

# Documentation Pages

- [Hardware configuration and schematics](./documentation/hardware.md)
- [Build and test instructions](./documentation/build_test.md)
- [Pipeline overview](./documentation/pipeline.md)
- [ReID train instruction](./documentation/train_instruction.md)
- [ReID network quantization for GAP](./documentation/quantization_instruction.md)
- [ReID network architecture and inference details](./documentation/network_inference.md)
- [Bluetooth LE protocol for users management](./documentation/ble_protocol.md)

# Papers

- 512KiB RAM Is Enough! Live Camera Face Recognition DNN on MCU: [link](http://openaccess.thecvf.com/content_ICCVW_2019/html/LPCV/Zemlyanikin_512KiB_RAM_Is_Enough_Live_Camera_Face_Recognition_DNN_on_ICCVW_2019_paper.html)

```
@InProceedings{Zemlyanikin_2019_ICCV,
author = {Zemlyanikin, Maxim and Smorkalov, Alexander and Khanova, Tatiana and Petrovicheva, Anna and Serebryakov, Grigory},
title = {512KiB RAM Is Enough! Live Camera Face Recognition DNN on MCU},
booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
month = {Oct},
year = {2019}
}
```

- GreenWaves press release and demostration video: [link](https://greenwaves-technologies.com/face_reid_on_gap8/)
