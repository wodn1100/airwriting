// AirWriting_ESP32_Serial.ino
// Serial (USB) streaming variant for easy testing
// - S1, S2: MPU6050 on I2C bus 0
// - S3: ICM20948 + AK09916 on I2C bus 1
// - Binary packet sent over Serial instead of WiFi/UDP

#include <Wire.h>

// ── Forward Declarations ──
struct SensorData6;
struct SensorData9;

void readMPU6050(uint8_t addr, SensorData6 &data);
void readICM20948(SensorData9 &data);

const int PEN_BTN_PIN = 15;

// Debounce state
uint8_t lastBtnState = 1;
uint8_t stableBtnState = 1;
uint8_t lastUnstableBtnState = 1;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 100;

// I2C bus pins
const int I2C0_SDA = 21;
const int I2C0_SCL = 22;
const int I2C1_SDA = 32;
const int I2C1_SCL = 33;

TwoWire *I2C_MPU;
TwoWire *I2C_ICM;

// IMU addresses per bus
const int ADDR_S1_MPU = 0x68;
const int ADDR_S2_MPU = 0x69;
const int ADDR_S3_ICM = 0x68;
const int ADDR_MAG = 0x0C;

// ── Serial command buffer ──
String serialCommandBuffer;

// ── Packet structure (identical to v3) ──
#pragma pack(push, 1)
struct SensorData6 {
  float ax, ay, az;
  float gx, gy, gz;
};

struct SensorData9 {
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
};

struct AirWritingPacketV3 {
  uint8_t header;
  uint32_t timestamp;
  SensorData6 s1;
  SensorData6 s2;
  SensorData9 s3;
  uint8_t button;
  uint8_t checksum;
  uint8_t footer;
};
#pragma pack(pop)

AirWritingPacketV3 packet;

// ══════════════════════════════════════════════
//  IMU Setup & Read (unchanged from v3)
// ══════════════════════════════════════════════

void setICMBank(uint8_t bank) {
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x7F);
  I2C_ICM->write(bank << 4);
  I2C_ICM->endTransmission(true);
}

void setupMPU6050(uint8_t addr) {
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x6B);
  I2C_MPU->write(0x01);
  I2C_MPU->endTransmission(true);

  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x1C);
  I2C_MPU->write(0x10);
  I2C_MPU->endTransmission(true);

  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x1B);
  I2C_MPU->write(0x10);
  I2C_MPU->endTransmission(true);
}

void setupICM20948() {
  Serial.println(">>> Initializing ICM20948 (Robust Mode) <<<");
  setICMBank(0);

  // 1. Device Reset
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x06); // PWR_MGMT_1 (ICM20948=0x06, NOT 0x6B!)
  I2C_ICM->write(0x80); // Reset
  I2C_ICM->endTransmission(true);
  delay(100);

  // 2. Wake Up
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x06); // PWR_MGMT_1 (ICM20948=0x06)
  I2C_ICM->write(0x01); // Auto select clock, SLEEP=0
  I2C_ICM->endTransmission(true);
  delay(50);

  // 3. Force Enable Accel & Gyro
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x07); // PWR_MGMT_2 (ICM20948=0x07, NOT 0x6C!)
  I2C_ICM->write(0x00); // 0 = Enable all axes
  I2C_ICM->endTransmission(true);
  delay(10);

  // 4. Disable I2C Master
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x03); // USER_CTRL
  I2C_ICM->write(0x00); // disable I2C master
  I2C_ICM->endTransmission(true);
  delay(10);

  // 5. Bypass Enable for Mag
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x0F); // INT_PIN_CFG
  I2C_ICM->write(0x02); // BYPASS_EN
  I2C_ICM->endTransmission(true);
  delay(10);

  // 6. Config Bank 2
  setICMBank(2);
  
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x14); // ACCEL_CONFIG (0x14 in bank 2)
  I2C_ICM->write(0x05); // +/- 4g
  I2C_ICM->endTransmission(true);

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x01); // GYRO_CONFIG_1
  I2C_ICM->write(0x13); // +/- 1000 dps, DLPF enabled
  I2C_ICM->endTransmission(true);

  setICMBank(0);

  // 7. AK09916 Magnetometer Setup (Continuous Mode 4 = 100Hz)
  // Bypass 모드가 활성화된 상태이므로, AK09916에 직접 I2C로 접근 가능
  
  // 7a. AK09916 소프트 리셋
  I2C_ICM->beginTransmission(ADDR_MAG);
  I2C_ICM->write(0x32); // CNTL3 (Soft Reset) — 주소 0x32
  I2C_ICM->write(0x01); // Reset
  I2C_ICM->endTransmission(true);
  delay(100);

  // 7b. Continuous Mode 4 (100Hz) 활성화
  I2C_ICM->beginTransmission(ADDR_MAG);
  I2C_ICM->write(0x31); // CNTL2
  I2C_ICM->write(0x08); // Mode 4 = 100Hz continuous measurement
  I2C_ICM->endTransmission(true);
  delay(10);

  Serial.println("AK09916 Magnetometer: Continuous Mode 100Hz enabled");
}

void readMPU6050(uint8_t addr, SensorData6 &data) {
  I2C_MPU->beginTransmission(addr);
  I2C_MPU->write(0x3B);
  I2C_MPU->endTransmission(false);
  I2C_MPU->requestFrom((int)addr, 14, (int)true);

  if (I2C_MPU->available() == 14) {
    int16_t ax = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t ay = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t az = (I2C_MPU->read() << 8 | I2C_MPU->read());
    I2C_MPU->read();
    I2C_MPU->read();
    int16_t gx = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t gy = (I2C_MPU->read() << 8 | I2C_MPU->read());
    int16_t gz = (I2C_MPU->read() << 8 | I2C_MPU->read());

    data.ax = ax * (9.81f / 4096.0f);
    data.ay = ay * (9.81f / 4096.0f);
    data.az = az * (9.81f / 4096.0f);
    data.gx = gx * ((PI / 180.0f) / 32.8f);
    data.gy = gy * ((PI / 180.0f) / 32.8f);
    data.gz = gz * ((PI / 180.0f) / 32.8f);
  } else {
    data.ax = data.ay = data.az = 0;
    data.gx = data.gy = data.gz = 0;
  }
}

void readICM20948(SensorData9 &data) {
  setICMBank(0); // MUST be in Bank 0 to read data registers!

  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x2D); // ACCEL_XOUT_H
  byte err = I2C_ICM->endTransmission(true); // MUST BE TRUE for some clones/modules!
  int bytes = I2C_ICM->requestFrom((int)ADDR_S3_ICM, 12, (int)true);

  if (bytes == 12) {
    int16_t ax = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t ay = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t az = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t gx = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t gy = (I2C_ICM->read() << 8 | I2C_ICM->read());
    int16_t gz = (I2C_ICM->read() << 8 | I2C_ICM->read());

    data.ax = ax * (9.81f / 8192.0f); // ±4g = 8192 LSB/g
    data.ay = ay * (9.81f / 8192.0f);
    data.az = az * (9.81f / 8192.0f);
    data.gx = gx * ((PI / 180.0f) / 32.8f); // 1000 dps
    data.gy = gy * ((PI / 180.0f) / 32.8f);
    data.gz = gz * ((PI / 180.0f) / 32.8f);
  } else {
    data.ax = data.ay = data.az = 0;
    data.gx = data.gy = data.gz = 0;
    Serial.printf("[Debug] ICM Read Failed! err=%d, requested 12, got %d\n", err, bytes);
  }

  I2C_ICM->beginTransmission(ADDR_MAG);
  I2C_ICM->write(0x11);
  I2C_ICM->endTransmission(false);
  I2C_ICM->requestFrom((int)ADDR_MAG, 7, (int)true);

  if (I2C_ICM->available() == 7) {
    int16_t mx = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    int16_t my = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    int16_t mz = (I2C_ICM->read() | (I2C_ICM->read() << 8));
    I2C_ICM->read();

    data.mx = mx * 0.15f;
    data.my = my * 0.15f;
    data.mz = mz * 0.15f;
  } else {
    data.mx = data.my = data.mz = 0;
  }
}

// ══════════════════════════════════════════════
//  Verbose I2C Debugger for ICM20948
// ══════════════════════════════════════════════
void debugWHOAMI() {
  Serial.println(">>> Checking ICM20948 WHO_AM_I (Register 0x00, Bank0) <<<");
  setICMBank(0);
  
  I2C_ICM->beginTransmission(ADDR_S3_ICM);
  I2C_ICM->write(0x00);
  byte error = I2C_ICM->endTransmission(false);
  Serial.printf("endTransmission returned: %d\n", error);
  
  int bytes = I2C_ICM->requestFrom((int)ADDR_S3_ICM, 1, (int)true);
  Serial.printf("requestFrom returned: %d bytes\n", bytes);
  
  if (I2C_ICM->available()) {
    byte who = I2C_ICM->read();
    Serial.printf("WHO_AM_I is: 0x%02X (Expected: 0xEA)\n", who);
  } else {
    Serial.println("Wire.available() == 0. failed to read!");
  }
}

// ══════════════════════════════════════════════
//  I2C Scanner
// ══════════════════════════════════════════════

void scanI2C(TwoWire *wire, const char *busName) {
  uint8_t error, address;
  int nDevices = 0;
  Serial.printf("Scanning %s...\n", busName);
  for (address = 1; address < 127; address++) {
    wire->beginTransmission(address);
    error = wire->endTransmission();
    if (error == 0) {
      Serial.printf("I2C device found at address 0x%02X\n", address);
      nDevices++;
    } else if (error == 4) {
      Serial.printf("Unknown error at address 0x%02X\n", address);
    }
  }
  if (nDevices == 0) {
    Serial.println("No I2C devices found.\n");
  } else {
    Serial.println("done\n");
  }
}

// ══════════════════════════════════════════════
//  Serial Command Handling 
// ══════════════════════════════════════════════

void processSerialCommand(String rawCommand) {
  String command = rawCommand;
  command.trim();
  if (command.length() == 0) {
    return;
  }

  String upper = command;
  upper.toUpperCase();

  if (upper == "PING") {
    Serial.println("PONG");
    return;
  }
  if (upper == "REBOOT") {
    Serial.println("Rebooting...");
    delay(200);
    ESP.restart();
    return;
  }
}

void handleSerialInput() {
  while (Serial.available() > 0) {
    char ch = static_cast<char>(Serial.read());
    if (ch == '\r') {
      continue;
    }
    if (ch == '\n') {
      processSerialCommand(serialCommandBuffer);
      serialCommandBuffer = "";
      continue;
    }
    if (serialCommandBuffer.length() < 255) {
      serialCommandBuffer += ch;
    }
  }
}

// ══════════════════════════════════════════════
//  Setup
// ══════════════════════════════════════════════

void setup() {
  Serial.begin(921600);
  delay(200);

  Serial.println("====================================");
  Serial.println("AirWriting ESP32 — IMU Only Mode");
  Serial.println("====================================");

  // Swap hardware I2C peripherals: ICM on 0, MPU on 1
  I2C_ICM = new TwoWire(0);
  I2C_MPU = new TwoWire(1);

  I2C_MPU->begin(I2C0_SDA, I2C0_SCL, 400000);
  I2C_ICM->begin(I2C1_SDA, I2C1_SCL, 400000);

  Serial.println("I2C Scanner - Checking Connections...");
  scanI2C(I2C_MPU, "Bus I2C_MPU (Pins 21/22)");
  scanI2C(I2C_ICM, "Bus I2C_ICM (Pins 32/33)");
  Serial.println("====================================");

  pinMode(PEN_BTN_PIN, INPUT_PULLUP);

  packet.header = 0xAA;
  packet.footer = 0x55;

  setupMPU6050(ADDR_S1_MPU);
  setupMPU6050(ADDR_S2_MPU);

  setupICM20948();

  // ICM20948 읽기 검증
  debugWHOAMI();
  
  SensorData9 testData = {};
  readICM20948(testData);
  Serial.printf("ICM20948 verify: ax=%.2f ay=%.2f az=%.2f\n", testData.ax, testData.ay, testData.az);
  if (testData.ax == 0 && testData.ay == 0 && testData.az == 0) {
    Serial.println("⚠️ ICM20948 returns zeros! Retrying setup...");
    delay(100);
    setupICM20948();
    delay(50);
    debugWHOAMI();
    readICM20948(testData);
    Serial.printf("ICM20948 retry: ax=%.2f ay=%.2f az=%.2f\n", testData.ax, testData.ay, testData.az);
  }

  Serial.println("Ready. Streaming binary packets over Serial.");
}

// ══════════════════════════════════════════════
//  Loop
// ══════════════════════════════════════════════

void loop() {
  // 호스트로부터 명령 수신 (OLED 업데이트 등)
  handleSerialInput();

  packet.timestamp = millis();

  // 센서 읽기
  readMPU6050(ADDR_S1_MPU, packet.s1);
  readMPU6050(ADDR_S2_MPU, packet.s2);
  
  readICM20948(packet.s3);

  // 버튼 디바운스
  uint8_t currentRawBtnState = digitalRead(PEN_BTN_PIN);

  if (currentRawBtnState != lastUnstableBtnState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (currentRawBtnState != stableBtnState) {
      stableBtnState = currentRawBtnState;
    }
  }
  lastUnstableBtnState = currentRawBtnState;

  packet.button = (stableBtnState == LOW) ? 1 : 0;
  lastBtnState = stableBtnState;

  // 체크섬 계산
  uint8_t *ptr = (uint8_t *)&packet;
  uint8_t cksum = 0;
  for (int i = 1; i < 90; i++) {
    cksum ^= ptr[i];
  }
  packet.checksum = cksum;

  // ★ Serial로 바이너리 패킷 전송 (UDP 대신)
  Serial.write(ptr, sizeof(AirWritingPacketV3));

  delay(10);
}
