EESchema Schematic File Version 2
LIBS:power
LIBS:device
LIBS:transistors
LIBS:conn
LIBS:linear
LIBS:regul
LIBS:74xx
LIBS:cmos4000
LIBS:adc-dac
LIBS:memory
LIBS:xilinx
LIBS:microcontrollers
LIBS:dsp
LIBS:microchip
LIBS:analog_switches
LIBS:motorola
LIBS:texas
LIBS:intel
LIBS:audio
LIBS:interface
LIBS:digital-audio
LIBS:philips
LIBS:display
LIBS:cypress
LIBS:siliconi
LIBS:opto
LIBS:atmel
LIBS:contrib
LIBS:valves
EELAYER 25 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L SW_PUSH SW1
U 1 1 5D53DC01
P 1000 700
F 0 "SW1" H 1150 810 50  0000 C CNN
F 1 "SW_PUSH" H 1000 620 50  0000 C CNN
F 2 "" H 1000 700 50  0000 C CNN
F 3 "" H 1000 700 50  0000 C CNN
	1    1000 700 
	1    0    0    -1  
$EndComp
$Comp
L R R1
U 1 1 5D53DD1A
P 1300 850
F 0 "R1" V 1380 850 50  0000 C CNN
F 1 "10k" V 1300 850 50  0000 C CNN
F 2 "" V 1230 850 50  0000 C CNN
F 3 "" H 1300 850 50  0000 C CNN
	1    1300 850 
	1    0    0    -1  
$EndComp
$Comp
L +1V8 #PWR?
U 1 1 5D53DE71
P 700 1000
F 0 "#PWR?" H 700 850 50  0001 C CNN
F 1 "+1V8" H 700 1140 50  0000 C CNN
F 2 "" H 700 1000 50  0000 C CNN
F 3 "" H 700 1000 50  0000 C CNN
	1    700  1000
	-1   0    0    1   
$EndComp
$Comp
L GND #PWR?
U 1 1 5D53DEA7
P 1300 1000
F 0 "#PWR?" H 1300 750 50  0001 C CNN
F 1 "GND" H 1300 850 50  0000 C CNN
F 2 "" H 1300 1000 50  0000 C CNN
F 3 "" H 1300 1000 50  0000 C CNN
	1    1300 1000
	1    0    0    -1  
$EndComp
Connection ~ 1300 700 
Wire Wire Line
	700  700  700  1000
Text GLabel 1750 900  3    60   Input ~ 0
GPIO
Wire Wire Line
	1750 900  1750 700 
Wire Wire Line
	1750 700  1300 700 
$EndSCHEMATC
