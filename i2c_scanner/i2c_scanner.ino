#include <Wire.h>

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // ICM20948이 연결되어 있는 I2C 1번 버스(32, 33) 스캔
  Wire.begin(32, 33, 400000); 
  Serial.println("\nI2C Scanner on pins 32(SDA) and 33(SCL)");
}

void loop() {
  byte error, address;
  int nDevices = 0;
  
  Serial.println("Scanning 32/33 pins...");
  
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    
    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address < 16) 
        Serial.print("0");
      Serial.print(address, HEX);
      Serial.println(" !");
      nDevices++;
    }
  }
  
  if (nDevices == 0) {
    Serial.println("No I2C devices found.\n");
  } else {
    Serial.println("done\n");
  }
  
  delay(3000);
}
