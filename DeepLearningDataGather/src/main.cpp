#include <Arduino.h>

/*==================================================*/
/*================ Library Includes ================*/
/*==================================================*/

#include <Wire.h>
#include "SparkFun_MMA8452Q.h"
#include <SPI.h>
#include <SD.h>
#include <ACROBOTIC_SSD1306.h>

const int chipSelect = 5;

/*==================================================*/
/*===================== Defines ====================*/
/*==================================================*/

#define I2C_CLK 21
#define I2C_DT 22

#define DIP_PIN_1 2
#define DIP_PIN_2 15

#define BUTTON_PIN 4

#define FILE_NAME "data.txt"

#define SENSOR_READ_INTERVAL 200 // times a second that the sensor reads data. 

/*==================================================*/
/*================ Global Variables ================*/
/*==================================================*/

MMA8452Q accel; // acceleration sensor
bool gatheringData = false;
unsigned long previousTime = 0;

File dataFile;

bool currentButtonState = false;
bool previousButtonState = false;

enum DataType { Walking, Running, Cycling, ClimbingStairs };
DataType dataType;


/*==================================================*/
/*============== Function Definitions ==============*/
/*==================================================*/

void WriteSensorValuesToFile(short, short, short, DataType);

/*==================================================*/
/*====================== Setup =====================*/
/*==================================================*/

void setup() {
  
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

  Serial.print("Initializing SD card...");

  // see if the card is present and can be initialized:
  if (!SD.begin(chipSelect)) 
  {
    Serial.println("Card failed, or not present");
    oled.clearDisplay();
    oled.setTextXY(0,0);
    oled.putString("Card failed");
    oled.setTextXY(1,0);
    oled.putString("or not present");
    while (1); // halt microcontroller
  }
  else
  {
    Serial.println("card initialized.");
  }

  dataFile = SD.open(FILE_NAME, FILE_WRITE);
  if (!dataFile)
  {
    Serial.print("error opening ");
    Serial.println(FILE_NAME);
    oled.clearDisplay();
    oled.setTextXY(0,0);
    oled.putString("error opening");
    oled.setTextXY(1,0);
    oled.putString(FILE_NAME);
    while (1); // halt microcontroller
  }  
  dataFile.close();

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
  
  accel.setScale(SCALE_4G); // set scale, can choose between: SCALE_2G - SCALE_4G - SCALE_8G

  delay(300);

  

  /*===================================================*/
  /*========== Setting Pin Modes Of Used Pins =========*/
  /*===================================================*/

  pinMode(DIP_PIN_1, INPUT_PULLDOWN);
  pinMode(DIP_PIN_2, INPUT_PULLDOWN);

  pinMode(BUTTON_PIN, INPUT_PULLDOWN);

  delay(300);

  /*===================================================*/
}

void loop() 
{
  unsigned long currentTime = millis(); // Get current time for consistant value throughout the loop

  if (previousTime + (1000/SENSOR_READ_INTERVAL) <= currentTime) // do things on sensor read interval
  {
    previousTime = currentTime;

    currentButtonState = digitalRead(BUTTON_PIN);

    if (previousButtonState && !currentButtonState) // button pressed toggle data gathering state
    {
      if (gatheringData)
      {
        Serial.println("Stopped gathering data");
        gatheringData = false;

        oled.clearDisplay();

        oled.setTextXY(3,5);              // Set cursor position, start of line 3
        String startString = "Project";  
        oled.putString(startString);

        oled.setTextXY(4,6);              // Set cursor position, start of line 4
        startString = "Deep";  
        oled.putString(startString);

        oled.setTextXY(5,4);              // Set cursor position, start of line 5
        startString = "Learning";  
        oled.putString(startString);

        // Close the file
        dataFile.flush();
        dataFile.close();
      }
      else
      {
        Serial.println("Started gathering data");
        gatheringData = true;

        // select data type according to dip switch positions
        bool dip1 = digitalRead(DIP_PIN_1);
        bool dip2 = digitalRead(DIP_PIN_2);
        
        String dataTypeString;
        int dataXLocation; // locatin to print on screen
        if (!dip1 && !dip2)
        {
          dataType = Walking;
          dataTypeString = "Walking";
          dataXLocation = 5;
        }
        else if (dip1 && !dip2)
        {
          dataType = Running;
          dataTypeString = "Running";
          dataXLocation = 5;
        }
        else if (!dip1 && dip2)
        {
          dataType = Cycling;
          dataTypeString = "Cycling";
          dataXLocation = 5
        }
        else if (dip1 && dip2)
        {
          dataType = ClimbingStairs;
          dataTypeString = "Stairs Climbing";
          dataXLocation = 1;
        }

        oled.clearDisplay();

        oled.setTextXY(3,3);              // Set cursor position, start of line 3
        String startString = "Gathering";  
        oled.putString(startString);

        oled.setTextXY(4,dataXLocation);  // Set cursor position, start of line 4 
        oled.putString(dataTypeString);

        oled.setTextXY(5,5);              // Set cursor position, start of line 5
        startString = "Data";  
        oled.putString(startString);

        // open the file
        dataFile = SD.open(FILE_NAME, FILE_WRITE);

        // send first line to file
        char buff[] = "Gathering Data:\n";
        dataFile.write(buff, sizeof(buff)-1);

        // wait half a second to stabalize and get doing activity
        previousTime += 500;
      }
    }
    previousButtonState = currentButtonState;
    

    if (gatheringData)
    {
      if (accel.available()) 
      {      
        accel.read(); // update accel values of sensor
        WriteSensorValuesToFile(accel.x, accel.y, accel.z, dataType); // write current values to sd
      }  
      else
      {
        WriteSensorValuesToFile(accel.x, accel.y, accel.z, dataType); // write previous values to sd to avoid missing datapoints
      }
      
    }  
  }
}


void WriteSensorValuesToFile(short x, short y, short z, DataType dt)
{
  int bufferSize = 22;
  char buffer[bufferSize];

  int length = snprintf(buffer, bufferSize, "%d,%d,%d,%d\n", x, y, z, (int)dt);

  // if the file is available, write to it:
  dataFile.write(buffer, length);
}
