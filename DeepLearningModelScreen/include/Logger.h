#pragma once
#include <SD.h>

class Logger
{
private:
    File dataFile;
    String fileName;
public:
    Logger(int chipSelect, String fileName);
    ~Logger();
    
    void Log(char* data, size_t length);
    void Log(String data);
    void Log(int type, int typeReference);

    void Start();
    void Stop();
};


