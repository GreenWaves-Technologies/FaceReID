Intro
=====

The project is aimed to create and deploy a smart doorbell application that would be highly energy-efficient. The main application pipeline consists of 4 stages: frame capture, face detection, face recognition and user interaction e.g. displaying message. In addition to regular face detection and re-identification loop, application provides functionality to add known people to the trusted list using already calculated face descriptors.

![](documentation/images/reid_pipeline.png)

Target Platforms
================

The demo is targeted for two boards Gapoc A and Gapuino. List of extra components is provided bellow.

For Gapuino board:
- HIGHMAX camera module
- Adafruit 2.8 TFT display with SPI interface
- PS/2 Keyboard

For Gapoc A board:
- Adafruit 2.8 TFT display with SPI interface
- Push button, 10kOm resistor and some wires
- Android-based smartphone

Documentation Pages
===================

- [Hardware configuration and schematics](./documentation/hardware.md)
- [Build and test instructions](./documentation/build_test.md)
- [Pipeline overview](./documentation/pipeline.md)
- [ReID network architecture and inference details](./documentation/network_inference.md)
- [Bluetooth LE protocol for users management](./documentation/ble_protocol.md)
