#include <TensorModel.h>
<<<<<<< HEAD
#include "Walking_Running_Model.h"
=======
#include "Walking_Running_Stairs_Cycling_Model_Copied_Stairs_Data.h"
>>>>>>> 976369175321d2b55f7e1c1e59d4a8e8dbcb7b75

TensorModel::TensorModel()
{
    static tflite::MicroErrorReporter micro_error_reporter;
    errorReporter = &micro_error_reporter;

<<<<<<< HEAD
    model = tflite::GetModel(Walking_Running_Model);
=======
    model = tflite::GetModel(Walking_Running_Stairs_Cycling_Model);
>>>>>>> 976369175321d2b55f7e1c1e59d4a8e8dbcb7b75
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        errorReporter->Report("Model version does nog match Schema");
        while(1);
    }
    
    static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
    micro_mutable_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::ops::micro::Register_FULLY_CONNECTED(),
        1, 9
    );

    micro_mutable_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_CONV_2D,
        tflite::ops::micro::Register_CONV_2D(),
        1, 5
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

void TensorModel::setModelInput(float *input)
{
    model_input->data.f = input;
}

int TensorModel::GetOutputSize()
{
    int outputSize = 1; // get amount of outputs
    for (size_t i = 0; i < model_output->dims->size; i++)
    {
        outputSize = outputSize * model_output->dims->data[i];
    }
    return outputSize;
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

float* TensorModel::predict()
{
    time_t startTime = micros();

    // test input on model
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) 
    {
        Serial.println("Invoke failed");
        errorReporter->Report("Invoke failed on input");
        return nullptr;
    }
    else
    {
        printModelOutput(micros() - startTime, model_output->data.f);
        return model_output->data.f;
    }
    
}

void TensorModel::printModelOutput(unsigned long computingTime, float *output)
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

TensorModel::~TensorModel()
{

}