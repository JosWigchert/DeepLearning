#include <Arduino.h>

#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

/*==================================================*/
/*================ Library Includes ================*/
/*==================================================*/

#include <Wire.h>
#include <SparkFun_MMA8452Q.h>
#include <SPI.h>
#include <ACROBOTIC_SSD1306.h>

#include "TensorModel.h"
#include "testArray.h"
#include "Logger.h"
/*==================================================*/
/*===================== Defines ====================*/
/*==================================================*/

#define ChipSelect 5 
#define LogFileName "log.txt" 

#define I2C_CLK 21
#define I2C_DT 22

#define DIP_PIN_1 2
#define DIP_PIN_2 15

#define BUTTON_PIN 4

#define SECOND 1000

#define SENSOR_READ_INTERVAL 50 // Read interval in Hz

#define INPUT_ARRAY_SIZE 300

#define DATA_TYPE_BUFFER_TIME .5
#define DATA_TYPE_BUFFER_SIZE 200

#define SCALE_X 1024
#define SCALE_Y 1024
#define SCALE_Z 1024

/*==================================================*/
/*================ Global Variables ================*/
/*==================================================*/

namespace {

  enum DataType { Walking, Running, Cycling, ClimbingStairs, Unknown };
  
  MMA8452Q accel; // acceleration sensor
  Logger* logger;
  DataType *dataTypeBuffer = new DataType[DATA_TYPE_BUFFER_SIZE];
  int dataTypeBufferIndex = 0;

  DataType dataType;
  DataType dataTypeReference;

  // unsigned long Time1 = 0;
  unsigned long Time2 = 0;
  float *input_array = new float[INPUT_ARRAY_SIZE];
  volatile float *input_array_buffer = new float[INPUT_ARRAY_SIZE];
  TensorModel *model;

  // hw_timer_t * timer = NULL;
  // portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

  size_t currentIndex = 0;

  bool loggingData = false;

  bool currentButtonState = false;
  bool previousButtonState = false;
}

float getHighest(float* array, int size)
{
  float highest = 0;
  for (size_t i = 0; i < size; i++)
  {
    if(array[i] > highest)
    {
      highest = array[i];
    }
  }
  return highest;
}

int getHighestIndex(float* array, int size)
{
  float highest = 0;
  int highestIndex = 0;
  for (size_t i = 0; i < size; i++)
  {
    if(array[i] > highest)
    {
      highest = array[i];
      highestIndex = i;
    }
  }
  return highestIndex;
}

void setActivityOnScreen(DataType dataType, float value)
{
  //Serial.print("Activity: ");
  //Serial.print(dataType);
  //Serial.print("  With value: ");
  //Serial.println(value);

  oled.setTextXY(0,0);
  oled.putString("Activity       ");
  oled.setTextXY(1,0);
  switch (dataType)
  {
  case Walking:
    oled.putString("Walking        ");
    break;
  case Running:
    oled.putString("Running        ");
    break;
  case Cycling:
    oled.putString("Cycling        ");
    break;
  case ClimbingStairs:
    oled.putString("Climbing Stairs");
    break;
  case Unknown:
    oled.putString("Unknown        ");
    break;
  default:
    break;
  }
  oled.setTextXY(2,0);
  oled.putString("With value     ");
  oled.setTextXY(3,0);
  oled.putFloat(value);
}

float normalize(float toScale, float scale)
{
  return toScale / scale;
}

void IRAM_ATTR readAccelSensor();

void setup() {
  Serial.println("Version 1.3 of Deeplearning Project");

  // Open serial communications and wait for port to open:
  Serial.begin(115200);
  while (!Serial);// wait for serial port to connect. Needed for native USB port only
  Wire.begin(I2C_DT, I2C_CLK, 40000); // Setting i2c bus

  /*===================================================*/
  /*=============== Setting OLED Screen ===============*/
  /*===================================================*/

  oled.init();                      // Initialze SSD1306 OLED display
  oled.setFont(font8x8);            // Set font type (default 8x8)
  oled.clearDisplay();              // Clear screen

  oled.setTextXY(3,5);              // Set cursor position, start of line 0
  String startString = "Project";  
  oled.putString(startString);

  oled.setTextXY(4,6);              // Set cursor position, start of line 0
  startString = "Deep";  
  oled.putString(startString);

  oled.setTextXY(5,4);              // Set cursor position, start of line 0
  startString = "Learning";  
  oled.putString(startString);

  /*===================================================*/
  /*=============== Setting SPI SD Card ===============*/
  /*===================================================*/

  logger = new Logger(ChipSelect, LogFileName);

  delay(300);

  /*===================================================*/
  /*========== Setting MMA8452 Accelerometer ==========*/
  /*===================================================*/

  /*  Able to pass other wire ports into the library
      Only possible on hardware with mutliple ports like the Teensy and Due
      Not possible on the Uno
  */
  if (accel.begin(Wire) == false) 
  {
    Serial.println("I2C peripheral Not Connected. Please check connections and read the hookup guide.");
    oled.clearDisplay();
    oled.setTextXY(0,0);
    oled.putString("MMA8452 Sensor");
    oled.setTextXY(1,0);
    oled.putString("Not working");
    while (1); // halt microcontroller
  }
  else
  {
    Serial.println("MMA8452 Initialization Successfull");
  }
  
  accel.setScale(SCALE_8G); // set scale, can choose between: SCALE_2G - SCALE_4G - SCALE_8G

  delay(300);
  
  /*===================================================*/
  /*========== Setting Pin Modes Of Used Pins =========*/
  /*===================================================*/

  pinMode(DIP_PIN_1, INPUT_PULLDOWN);
  pinMode(DIP_PIN_2, INPUT_PULLDOWN);

  pinMode(BUTTON_PIN, INPUT_PULLDOWN);

  delay(300);

  /*===================================================*/
  /*============= Setting TensorFlow Model ============*/
  /*===================================================*/

  model = new TensorModel();

  Serial.println("Calculating model");
  for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
  {
    input_array_buffer[i] = 0;
    input_array[i] = 0;
  }
  model->model_input->data.f = input_array;

  /*===================================================*/
}

// void setup() {
//   Serial.begin(115200);
//   while (!Serial);// wait for serial port to connect. Needed for native USB port only

//   logger = new Logger(ChipSelect, LogFileName);

//   Serial.println("Version 1.3 of Deeplearning Project");

//   Wire.begin(I2C_DT, I2C_CLK, 40000); // Setting i2c bus

//   /*===================================================*/
//   /*=============== Setting OLED Screen ===============*/
//   /*===================================================*/

//   oled.init();                      // Initialze SSD1306 OLED display
//   oled.setFont(font8x8);            // Set font type (default 8x8)
//   oled.clearDisplay();              // Clear screen

//   oled.setTextXY(3,5);              // Set cursor position, start of line 0
//   String startString = "Project";  
//   oled.putString(startString);

//   oled.setTextXY(4,6);              // Set cursor position, start of line 0
//   startString = "Deep";  
//   oled.putString(startString);

//   oled.setTextXY(5,4);              // Set cursor position, start of line 0
//   startString = "Learning";  
//   oled.putString(startString);

//   /*===================================================*/
//   /*========== Setting MMA8452 Accelerometer ==========*/
//   /*===================================================*/

//   /*  Able to pass other wire ports into the library
//       Only possible on hardware with mutliple ports like the Teensy and Due
//       Not possible on the Uno
//   */
//   if (accel.begin(Wire) == false) 
//   {
//     Serial.println("I2C peripheral Not Connected. Please check connections and read the hookup guide.");
//     oled.clearDisplay();
//     oled.setTextXY(0,0);
//     oled.putString("MMA8452 Sensor");
//     oled.setTextXY(1,0);
//     oled.putString("Not working");
//    // while (1); // halt microcontroller
//   }
//   else
//   {
//     Serial.println("MMA8452 Initialization Successfull");
//   }
  
//   accel.setScale(SCALE_8G); // set scale, can choose between: SCALE_2G - SCALE_4G - SCALE_8G

//   /*===================================================*/
//   /*========== Setting Pin Modes Of Used Pins =========*/
//   /*===================================================*/

//   pinMode(DIP_PIN_1, INPUT_PULLDOWN);
//   pinMode(DIP_PIN_2, INPUT_PULLDOWN);

//   pinMode(BUTTON_PIN, INPUT_PULLDOWN);

//   /*===================================================*/
//   /*============= Setting TensorFlow Model ============*/
//   /*===================================================*/

//   model = new TensorModel();
  
//   oled.clearDisplay();

//   Serial.println("Calculating model");
//   for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
//   {
//     input_array_buffer[i] = 0;
//     input_array[i] = 0;
//   }
//   model->model_input->data.f = input_array;
// }

void GatherData()
{
  if (accel.available()) 
  {      
    accel.read(); // update accel values of sensor
    input_array_buffer[currentIndex] = normalize(accel.x, 1024);
    input_array_buffer[currentIndex+1] = normalize(accel.y, 1024);
    input_array_buffer[currentIndex+2] = normalize(accel.z, 1024);

    currentIndex = currentIndex + 3;
    dataTypeBufferIndex++;
  }
}

void Calculate()
{
  for (size_t i = currentIndex; i < INPUT_ARRAY_SIZE; i++)
  {
    input_array[i-currentIndex] =  input_array_buffer[i];
  }

  for (size_t i = 0; i < currentIndex; i++)
  {
    input_array[i+(INPUT_ARRAY_SIZE - currentIndex)] =  input_array_buffer[i];
  }

  //Serial.println("Calculating model");
  float *output = model->predict();
  //Serial.println("Done calculating");
  int highestOutputIndex = getHighestIndex(output, model->GetOutputSize());

  //if (output[highestOutputIndex] > 0)
  {
    dataType = (DataType)highestOutputIndex;  
  }
  // else
  // {
  //   dataType = Unknown;
  // }
  dataTypeBuffer[dataTypeBufferIndex] = dataType;
  dataTypeBufferIndex++;

  //setActivityOnScreen(dataType, output[highestOutputIndex]);

  if (currentIndex >= INPUT_ARRAY_SIZE)
  {
    currentIndex = 0;
  }
}

void LogActivity(DataType datatype, DataType datatypeReference)
{
  // int walking = 0;
  // int running = 0;
  // int cycling = 0;
  // int stairs = 0;

  // for (size_t i = 0; i < DATA_TYPE_BUFFER_SIZE; i++)
  // {
  //   switch (dataTypeBuffer[i])
  //   {
  //   case Walking:
  //     walking++;
  //     break;
  //   case Running:
  //     running++;
  //     break;
  //   case Cycling:
  //     cycling++;
  //     break;
  //   case ClimbingStairs:
  //     stairs++;
  //     break;
  //   default:
  //     break;
  //   }
  // }

  // int maxValue = max(max(walking, running), max(cycling, stairs));
  // Serial.print("walking: ");
  // Serial.println(walking);
  // Serial.print("running: ");
  // Serial.println(running);
  // Serial.print("cycling: ");
  // Serial.println(cycling);
  // Serial.print("stairs: ");
  // Serial.println(stairs);

  // Serial.print("Max value: ");
  // Serial.println(maxValue);
  //float percentage = (float)maxValue / (float)(walking + running + cycling + stairs);

  logger->Log(datatype, datatypeReference);
  // if (maxValue == walking)
  // {
  //   logger->Log(Walking, percentage);
  // }
  // else if (maxValue == running)
  // {
  //   logger->Log(Running, percentage);
  // }
  // else if (maxValue == cycling)
  // {
  //   logger->Log(Cycling, percentage);
  // }
  // else if (maxValue == stairs)
  // {
  //   logger->Log(ClimbingStairs, percentage);
  // }
}

void loop() 
{
  unsigned long CurrentTime = millis();

  if (dataTypeBufferIndex == DATA_TYPE_BUFFER_SIZE)
  {
    dataTypeBufferIndex = 0;
    oled.clearDisplay();
    oled.setTextXY(0,3);
    oled.putString("Predicted");
    oled.setTextXY(1,4);
    oled.putString("Activity"); 
    switch (dataType)
    {
    case Walking:
      oled.setTextXY(4,4);
      oled.putString("Walking");
      break;
    case Running:
      oled.setTextXY(4,4);
      oled.putString("Running");
      break;
    case Cycling:
      oled.setTextXY(4,4);
      oled.putString("Cycling");
      break;
    case ClimbingStairs:
      oled.setTextXY(4,4);
      oled.putString("Cycling");
      break;
    
    default:
      break;
    }
  }

  if (Time2 + (1000 / SENSOR_READ_INTERVAL) < CurrentTime)
  {
    Serial.print("Starting cycle: ");
    Serial.println(millis());
    Time2 = CurrentTime;

    GatherData();
    Serial.print("Done gathering data: ");
    Serial.println(millis());

    Calculate();
    Serial.print("Done Calculating: ");
    Serial.println(millis());

    if (loggingData)
    {
      LogActivity(dataType, dataTypeReference);
      Serial.print("Done Logging: ");
      Serial.println(millis());
    } 

    currentButtonState = digitalRead(BUTTON_PIN);

    if (previousButtonState && !currentButtonState) // button pressed toggle data gathering state
    {
      if (loggingData)
      {
        Serial.println("Stopped logging data");
        loggingData = false;

        // Close the file
        logger->Stop();
      }
      else
      {
        Serial.println("Started logging data");

        // select data type according to dip switch positions
        bool dip1 = digitalRead(DIP_PIN_1);
        bool dip2 = digitalRead(DIP_PIN_2);
        Serial.println(dip1);
        Serial.println(dip2);
        if (!dip1 && !dip2)
        {
          dataTypeReference = Walking;
        }
        else if (dip1 && !dip2)
        {
          dataTypeReference = Running;
        }
        else if (!dip1 && dip2)
        {
          dataTypeReference = Cycling;
        }
        else if (dip1 && dip2)
        {
          dataTypeReference = ClimbingStairs;
        }
        
        loggingData = true;

        logger->Start();
        logger->Log(String("Log session started\n"));
      }
    }
    previousButtonState = currentButtonState;
    Serial.print("Done Cycle: ");
    Serial.println(millis());
    Serial.println();
  }
}
