#include <Arduino.h>

#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

#include "deeplearningModel.h"

#define DEBUG 1
#define INPUT_ARRAY_SIZE 300

#define MEAN_X -20.85741800
#define MEAN_Y -525.50149844
#define MEAN_Z 43.7810307

#define SCALE_X 338.02603217
#define SCALE_Y 489.93786421
#define SCALE_Z 256.92307326

void printModelOutput(unsigned long computingTime, float *output);

namespace {
  tflite::ErrorReporter* errorReporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  constexpr size_t kTensorArenaSize = 21*1024;
  uint8_t tensorArena[kTensorArenaSize];

  TaskHandle_t modelTaskHandle;
  
  unsigned long Time = 0;
  unsigned long Time2 = 0;
  float input_array[INPUT_ARRAY_SIZE];
}

void calculateActivity(void * parameter)
{
  while (1)
  {
    
    delay(1);
    break;
  }
  
  vTaskDelete(NULL);
}

void setup() {
  Serial.begin(115200);
  #if DEBUG
    //while (!Serial);
  #endif
  
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

  #if DEBUG
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

  #endif

  xTaskCreatePinnedToCore(
      calculateActivity, /* Function to implement the task */
      "CalculateActivity", /* Name of the task */
      100000,  /* Stack size in words */
      NULL,  /* Task input parameter */
      0,  /* Priority of the task */
      &modelTaskHandle,  /* Task handle. */
      0); /* Core where the task should run */

  model_input->data.f = input_array;
}

void loop() {
  if (Time + 450 <= millis())
  {
    Time = millis();

    
  }
  if (Time2 + 1000 < millis())
  {
    Time2 = millis();

    for (size_t i = 0; i < INPUT_ARRAY_SIZE; i++)
    {
      if (i % 2 == 0 && i != 0)
      {
        delayMicroseconds(5); // without it goes in error
        input_array[i] =  (((float)random(-2048, 2047) - MEAN_Y) / SCALE_Y);
      }
      else if (i % 2 == 0 && i != 0)
      {
        delayMicroseconds(5); // without it goes in error
        input_array[i] =  (((float)random(-2048, 2047) - MEAN_Z) / SCALE_Z);
      }
      else
      {
        delayMicroseconds(5); // without it goes in error
        input_array[i] =  (((float)random(-2048, 2047) - MEAN_X) / SCALE_X);
      }
    }

    time_t startTime = micros();

    // test input on model
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) 
    {
      Serial.println("Invoke failed");
      errorReporter->Report("Invoke failed on input");
    }
    else
    {
      printModelOutput(micros() - startTime, model_output->data.f);
    }
  }
}

void printModelOutput(unsigned long computingTime, float *output)
{
  Serial.println();
  Serial.print("Computing time (us): ");
  Serial.println(computingTime);
  
  int outputSize = 1; // get amount of outputs
  for (size_t i = 0; i < model_output->dims->size; i++)
  {
    outputSize = outputSize * model_output->dims->data[i];
  }
  
  if (outputSize >= 1)
  {
    Serial.print("Walking: ");
    Serial.println(output[0]);
  }
  if (outputSize >= 2)
  {
    Serial.print("Running: ");
    Serial.println(output[1]);
  }
  if (outputSize >= 3)
  {
    Serial.print("Cycling: ");
    Serial.println(output[2]);
  }
  if (outputSize >= 4)
  {
    Serial.print("Stairs: ");
    Serial.println(output[3]);
  }
}