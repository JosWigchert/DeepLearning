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
#include "SparkFun_MMA8452Q.h"
#include <SPI.h>
#include <ACROBOTIC_SSD1306.h>
#include "TensorModel.h"

/*==================================================*/
/*===================== Defines ====================*/
/*==================================================*/

#define I2C_CLK 21
#define I2C_DT 22

#define DIP_PIN_1 2
#define DIP_PIN_2 15

#define BUTTON_PIN 4

#define SENSOR_READ_INTERVAL 50 // Read interval in Hz

#define INPUT_ARRAY_SIZE 400

#define MEAN_X -20.85741800
#define MEAN_Y -525.50149844
#define MEAN_Z 43.7810307

#define SCALE_X 338.02603217
#define SCALE_Y 489.93786421
#define SCALE_Z 256.92307326

/*==================================================*/
/*================ Global Variables ================*/
/*==================================================*/

namespace {

  MMA8452Q accel; // acceleration sensor

  enum DataType { Walking = 0, Running, Cycling, ClimbingStairs, Unknown };
  DataType dataType;

  unsigned long Time1 = 0;
  unsigned long Time2 = 0;
  float *input_array = new float[INPUT_ARRAY_SIZE];
  volatile float *input_array_buffer = new float[INPUT_ARRAY_SIZE];
  TensorModel *model;

  int currentIndex = 0;

  hw_timer_t * timer = NULL;
  portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;
}

void IRAM_ATTR readAccelSensor();

void setup() {
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
   // while (1); // halt microcontroller
  }
  else
  {
    Serial.println("MMA8452 Initialization Successfull");
  }
  
  accel.setScale(SCALE_8G); // set scale, can choose between: SCALE_2G - SCALE_4G - SCALE_8G

  delay(300);

  model = new TensorModel();
  model->setModelInput(input_array);

  //timer = timerBegin(0, 80, true);
  //timerAttachInterrupt(timer, &readAccelSensor, true);
  //timerAlarmWrite(timer, 20000, true);
  //timerAlarmEnable(timer);

  delay(3000);
}

float normalize(float toScale, float mean, float scale)
{
  return (toScale - mean) / scale;
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

void IRAM_ATTR readAccelSensor()
{
  portENTER_CRITICAL_ISR(&timerMux);

  Serial.println("gathering data");
  Serial.println(millis());
  Serial.println();


  if (accel.available()) 
  {      
    accel.read(); // update accel values of sensor
    //delayMicroseconds(1); // without it goes in error
    input_array_buffer[currentIndex] = .1; // normalize(accel.x, MEAN_X, SCALE_X);
    //delayMicroseconds(1); // without it goes in error 
    input_array_buffer[currentIndex+1] = .1; // normalize(accel.y, MEAN_Y, SCALE_Y);
    //delayMicroseconds(1); // without it goes in error
    input_array_buffer[currentIndex+2] = .1; // normalize(accel.z, MEAN_Z, SCALE_Z);
    //delayMicroseconds(1); // without it goes in error

    currentIndex = currentIndex + 3;
    if (currentIndex >= 300)
    {
      currentIndex = 0;
    }
  }

  portEXIT_CRITICAL_ISR(&timerMux);
}

void setActivityOnScreen(DataType dataType, float value)
{
  Serial.print("Activity: ");
  Serial.print(dataType);
  Serial.print("  With value: ");
  Serial.println(value);

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

void loop() 
{
  unsigned long CurrentTime = millis();

  Serial.println("gathering data");
  Serial.println(millis());
  Serial.println();


  if (accel.available() && Time1) 
  {      
    accel.read(); // update accel values of sensor
    //delayMicroseconds(1); // without it goes in error
    input_array_buffer[currentIndex] = .1; // normalize(accel.x, MEAN_X, SCALE_X);
    //delayMicroseconds(1); // without it goes in error 
    input_array_buffer[currentIndex+1] = .1; // normalize(accel.y, MEAN_Y, SCALE_Y);
    //delayMicroseconds(1); // without it goes in error
    input_array_buffer[currentIndex+2] = .1; // normalize(accel.z, MEAN_Z, SCALE_Z);
    //delayMicroseconds(1); // without it goes in error

    currentIndex = currentIndex + 3;
    if (currentIndex >= 300)
    {
      currentIndex = 0;
    }
  }
  if (Time2 + 100 < CurrentTime)
  {
    Time2 = CurrentTime;

    Serial.println("Calculating model");

    time_t t = micros();

    //timerAlarmDisable(timer);
    //portENTER_CRITICAL(&timerMux);
    for (size_t i = currentIndex; i < INPUT_ARRAY_SIZE; i++)
    {
      input_array[i-currentIndex] =  input_array_buffer[i];
      delayMicroseconds(5); // without it goes in error
    }

    for (size_t i = 0; i < currentIndex; i++)
    {
      input_array[i+(INPUT_ARRAY_SIZE - currentIndex)] =  input_array_buffer[i];
      delayMicroseconds(5); // without it goes in error
    }
    //portEXIT_CRITICAL(&timerMux);
    //timerAlarmEnable(timer);
    
    float *output = model->predict();

    Serial.println("Done calculating");
    Serial.print("Calculating took: ");
    Serial.print(micros()-t);
    Serial.println(" us");

    int highestOutputIndex = getHighestIndex(output, model->getOutputSize());

    //if (output[highestOutputIndex] >)
    // {
      dataType = (DataType)highestOutputIndex;  
    //}
    //else
    // {
    //  dataType = Unknown;
    // }
    

    setActivityOnScreen(dataType, output[highestOutputIndex]);
  }
}