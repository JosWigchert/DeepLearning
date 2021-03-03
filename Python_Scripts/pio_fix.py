import os


uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])

f = open(uppath(__file__, 2) + '\\DeepLearningDataGather\\.pio\\libdeps\\fm-devkit\\SD\\src\\utility\\Sd2PinMap.h', 'r')

lines = f.readlines()

for i in range(len(lines)-1):
    if ('#error Architecture or board not supported.' in lines[i] and '//#error Architecture or board not supported.' not in lines[i]):
        lines[i] = '//#error Architecture or board not supported.\n#include "Arduino.h"\nuint8_t const SS_PIN = 15;\nuint8_t const MOSI_PIN = 23;\nuint8_t const MISO_PIN = 19;\nuint8_t const SCK_PIN = 18;\n'

f.close

f = open(uppath(__file__, 2) + '\\DeepLearningDataGather\\.pio\\libdeps\\fm-devkit\\SD\\src\\utility\\Sd2PinMap.h', 'w')
f.writelines(lines)
f.close()


f = open(uppath(__file__, 2) + '\\DeepLearningDataGather\\.pio\\libdeps\\fm-devkit\\ACROBOTIC SSD1306\\ACROBOTIC_SSD1306.cpp', 'r')

lines = f.readlines()

for i in range(len(lines)-1):
    if ('void ACROBOTIC_SSD1306::setFont(const uint8_t* font, bool inverse=false)' in lines[i]):
        lines[i] = 'void ACROBOTIC_SSD1306::setFont(const uint8_t* font, bool inverse)'
    elif ('setFont(font8x8);' in lines[i]):
        lines[i] = 'setFont(font8x8, false);'

f.close

f = open(uppath(__file__, 2) + '\\DeepLearningDataGather\\.pio\\libdeps\\fm-devkit\\ACROBOTIC SSD1306\\ACROBOTIC_SSD1306.cpp', 'w')
f.writelines(lines)
f.close()