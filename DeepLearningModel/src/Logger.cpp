#include "Logger.h"
#include <Arduino.h>

Logger::Logger(int chipSelect, String fileName)
{
    this->fileName = fileName;

    /*===================================================*/
    /*=============== Setting SPI SD Card ===============*/
    /*===================================================*/

    Serial.print("Initializing SD card...");

    // see if the card is present and can be initialized:
    if (!SD.begin(chipSelect)) 
    {
        Serial.println("Card failed, or not present");
        while (1); // halt microcontroller
    }
    else
    {
        Serial.println("card initialized.");
    }

    dataFile = SD.open(fileName, FILE_WRITE);
    if (!dataFile)
    {
        Serial.print("error opening ");
        Serial.println(fileName);
        while (1); // halt microcontroller
    }  
    dataFile.close();
}

Logger::~Logger()
{
    Stop();
}

void Logger::Log(char* data, size_t length)
{
    dataFile.write(data, length);
}

void Logger::Log(String data)
{
    dataFile.write(data.c_str(), data.length());
}

void Logger::Log(int type, int typeReference)
{
    Serial.print(type);
    Serial.print(";");
    Serial.println(typeReference);

    int bufferSize = 64;
    char buffer[bufferSize];

    int length = snprintf(buffer, bufferSize, "%d;%d\n", type, typeReference);

    // if the file is available, write to it:
    dataFile.write(buffer, length);
}

void Logger::Start()
{
    dataFile = SD.open(fileName, FILE_WRITE);
}

void Logger::Stop()
{
    if (dataFile)
    {
        dataFile.flush();
        dataFile.close();
    }
}