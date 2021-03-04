#pragma once

#include <Arduino.h>

#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

#include "deeplearningModel.h"

class TensorModel
{
public: // constructors and deconstructors
    TensorModel();
    virtual ~TensorModel();

public: // public methods
    float* predict();
    void printModelIO();
    void setModelInput(float* input);

private: // private methods
    

// variables
private:
    tflite::ErrorReporter* errorReporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* model_input = nullptr;
    TfLiteTensor* model_output = nullptr;

    static const uint32_t kTensorArenaSize = 21*1024;
    uint8_t tensorArena[kTensorArenaSize];

};

#include <TensorModel.h>

TensorModel::TensorModel()
{
    static tflite::MicroErrorReporter micro_error_reporter;
    errorReporter = &micro_error_reporter;

    model = tflite::GetModel(deeplearningModel);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        errorReporter->Report("Model version does nog match Schema");
        while(1);
    }
    
    static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
    micro_mutable_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::ops::micro::Register_FULLY_CONNECTED(),
        9, 9
    );

    micro_mutable_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_CONV_2D,
        tflite::ops::micro::Register_CONV_2D(),
        5, 5
    );

    micro_mutable_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_RESHAPE,
        tflite::ops::micro::Register_RESHAPE(),
        1, 1
    );
    
    micro_mutable_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_SOFTMAX,
        tflite::ops::micro::Register_SOFTMAX(),
        1, 1
    );

    static tflite::MicroInterpreter static_interpreter(
        model, micro_mutable_op_resolver, tensorArena, kTensorArenaSize, errorReporter
    );

    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        errorReporter->Report("AllocateTensors() failed");
        while(1);
    }
    
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    printModelIO();
}

void TensorModel::printModelIO()
{
    Serial.print("setup() running on core ");
    Serial.println(xPortGetCoreID());

    Serial.println("Model Input");
    Serial.print("Number of dimentions: ");
    int size = model_input->dims->size;
    
    Serial.println(size);
    for (size_t i = 0; i < size; i++)
    {
        Serial.print("Dim ");
        Serial.print(i);
        Serial.print(" size: ");
        Serial.println(model_input->dims->data[i]);
    }
    Serial.print("Input type: ");
    Serial.println(model_input->type);


    Serial.println("Model Output");
    Serial.print("Number of dimentions: ");
    size = model_output->dims->size;
    
    Serial.println(size);
    for (size_t i = 0; i < size; i++)
    {
        Serial.print("Dim ");
        Serial.print(i);
        Serial.print(" size: ");
        Serial.println(model_output->dims->data[i]);
    }
    Serial.print("Output type: ");
    Serial.println(model_output->type);
}

TensorModel::~TensorModel()
{

}