#pragma once

#include <Arduino.h>

#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

class TensorModel
{
public: // constructors and deconstructors
    TensorModel();
    virtual ~TensorModel();

public: // public methods
    float* predict();
    void setModelInput(float* input);
    

private: // private methods
    void printModelIO();
    void printModelOutput(unsigned long computingTime, float *output);

// variables
private:
    tflite::ErrorReporter* errorReporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    
    TfLiteTensor* model_output = nullptr;

    static const uint32_t kTensorArenaSize = 21*1024;
    uint8_t tensorArena[kTensorArenaSize];
public:
TfLiteTensor* model_input = nullptr;
};

