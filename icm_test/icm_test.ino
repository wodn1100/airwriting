#include <Wire.h>

// ★ 주소를 여기서 바꿀 수 있음 (AD0=GND→0x68, AD0=VCC→0x69)
const uint8_t ICM_ADDR = 0x68;
const int SDA_PIN = 32;
const int SCL_PIN = 33;

void setBank(uint8_t bank) {
  Wire.beginTransmission(ICM_ADDR);
  Wire.write(0x7F);
  Wire.write(bank << 4);
  Wire.endTransmission();
  delay(1);
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(ICM_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  Wire.requestFrom(ICM_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ICM_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
  delay(2);
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  
  Serial.println("\n============================================");
  Serial.println("  ICM20948 CLEAN INIT TEST v4");
  Serial.println("============================================");
  
  // ═══ 100kHz로 안정적으로 시작 ═══
  Wire.begin(SDA_PIN, SCL_PIN, 100000);
  delay(200);
  
  // ═══ STEP 1: I2C Scan (0x68 & 0x69 둘 다 체크) ═══
  Serial.println("\n--- STEP 1: I2C Scan ---");
  for (uint8_t addr = 0x68; addr <= 0x69; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    Serial.printf("  0x%02X → %s\n", addr, err == 0 ? "FOUND ✅" : "not found");
  }
  
  // ═══ STEP 2: WHO_AM_I ═══
  Serial.println("\n--- STEP 2: WHO_AM_I ---");
  setBank(0);
  uint8_t who = readReg(0x00);
  Serial.printf("  WHO_AM_I = 0x%02X (expect 0xEA)\n", who);
  if (who != 0xEA) {
    Serial.println("  ❌ ICM20948 not responding! Check wiring.");
    Serial.println("  → AD0=GND이면 0x68, AD0=VCC이면 0x69");
    while(1) delay(1000);
  }
  
  // ═══ STEP 3: 소프트 리셋 ═══
  Serial.println("\n--- STEP 3: Soft Reset ---");
  setBank(0);
  writeReg(0x06, 0x80);  // ★ ICM20948 PWR_MGMT_1 = 0x06 (NOT 0x6B!)
  delay(100);  // 리셋 완료 대기
  
  // 리셋 후 다시 WHO_AM_I 확인
  setBank(0);
  who = readReg(0x00);
  Serial.printf("  After reset WHO_AM_I = 0x%02X\n", who);
  
  // 리셋 후 기본값 확인
  uint8_t pwr1_default = readReg(0x06);
  Serial.printf("  PWR_MGMT_1 after reset = 0x%02X (expect 0x41=sleep)\n", pwr1_default);
  
  // ═══ STEP 4: Wake Up (SLEEP 비트 해제) ═══
  Serial.println("\n--- STEP 4: Wake Up ---");
  setBank(0);
  writeReg(0x06, 0x01);  // ★ 0x06! CLKSEL=1 (auto), SLEEP=0
  delay(50);
  
  uint8_t pwr1 = readReg(0x06);
  Serial.printf("  PWR_MGMT_1 = 0x%02X\n", pwr1);
  
  // 0x01이 아니면 0x00으로 시도
  if (pwr1 != 0x01 && pwr1 != 0x00) {
    Serial.println("  Trying 0x00 (internal osc)...");
    writeReg(0x06, 0x00);
    delay(50);
    pwr1 = readReg(0x06);
    Serial.printf("  PWR_MGMT_1 = 0x%02X\n", pwr1);
  }
  
  // ═══ STEP 5: 모든 축 활성화 ═══
  Serial.println("\n--- STEP 5: Enable All Axes ---");
  setBank(0);
  writeReg(0x07, 0x00);  // ★ 0x07! 모든 가속도/자이로 축 ON
  delay(50);
  Serial.printf("  PWR_MGMT_2 = 0x%02X (expect 0x00)\n", readReg(0x07));
  
  // ═══ STEP 6: Bank 2에서 센서 설정 ═══
  Serial.println("\n--- STEP 6: Sensor Config (Bank 2) ---");
  setBank(2);
  
  // 자이로 설정: DLPF 활성, ±250dps
  writeReg(0x01, 0x01);  // GYRO_CONFIG_1: DLPF_EN=1, FS=0(±250), DLPFCFG=0
  delay(10);
  Serial.printf("  GYRO_CONFIG_1 = 0x%02X (expect 0x01)\n", readReg(0x01));
  
  // 자이로 샘플레이트 = 1.1kHz / (1+10) = 100Hz
  writeReg(0x00, 0x0A);  // GYRO_SMPLRT_DIV = 10
  delay(10);
  Serial.printf("  GYRO_SMPLRT_DIV = 0x%02X\n", readReg(0x00));
  
  // 가속도 설정: DLPF 활성, ±2g
  writeReg(0x14, 0x01);  // ACCEL_CONFIG: DLPF_EN=1, FS=0(±2g), DLPFCFG=0
  delay(10);
  Serial.printf("  ACCEL_CONFIG = 0x%02X (expect 0x01)\n", readReg(0x14));
  
  // 가속도 샘플레이트 = 1.125kHz / (1+10) ≈ 102Hz
  writeReg(0x10, 0x00);  // ACCEL_SMPLRT_DIV_1 (MSB)
  writeReg(0x11, 0x0A);  // ACCEL_SMPLRT_DIV_2 (LSB) = 10
  delay(10);
  Serial.printf("  ACCEL_SMPLRT_DIV = 0x%02X 0x%02X\n", readReg(0x10), readReg(0x11));
  
  // ═══ STEP 7: Bank 0으로 복귀 & INT 설정 ═══
  Serial.println("\n--- STEP 7: INT Config ---");
  setBank(0);
  writeReg(0x0F, 0x01);  // INT_PIN_CFG: bypass mode (자기계 직접접근)
  delay(10);
  writeReg(0x10, 0x01);  // INT_ENABLE_1: RAW_DATA_RDY_EN
  delay(10);
  Serial.printf("  INT_PIN_CFG = 0x%02X\n", readReg(0x0F));
  Serial.printf("  INT_ENABLE_1 = 0x%02X\n", readReg(0x10));
  
  // ═══ STEP 8: LP_CONFIG 확인 ═══
  uint8_t lp = readReg(0x05);
  Serial.printf("  LP_CONFIG = 0x%02X", lp);
  if (lp & 0x20) {
    Serial.println(" → ACCEL in duty cycle, disabling...");
    writeReg(0x05, lp & 0x0F);  // continuous mode
    delay(10);
    Serial.printf("  LP_CONFIG = 0x%02X\n", readReg(0x05));
  } else {
    Serial.println(" → OK (continuous mode)");
  }
  
  // ═══ STEP 9: 2초 대기 후 데이터 읽기 ═══
  Serial.println("\n--- STEP 9: Wait 2s then read data ---");
  delay(2000);
  
  bool anyNonZero = false;
  for (int i = 0; i < 20; i++) {
    setBank(0);
    uint8_t status = readReg(0x1A);  // INT_STATUS_1
    
    Wire.beginTransmission(ICM_ADDR);
    Wire.write(0x2D);  // ACCEL_XOUT_H
    Wire.endTransmission(false);
    Wire.requestFrom(ICM_ADDR, (uint8_t)12);
    
    if (Wire.available() >= 12) {
      int16_t ax = (Wire.read() << 8) | Wire.read();
      int16_t ay = (Wire.read() << 8) | Wire.read();
      int16_t az = (Wire.read() << 8) | Wire.read();
      int16_t gx = (Wire.read() << 8) | Wire.read();
      int16_t gy = (Wire.read() << 8) | Wire.read();
      int16_t gz = (Wire.read() << 8) | Wire.read();
      
      Serial.printf("[%2d] ST=0x%02X A:%6d %6d %6d G:%6d %6d %6d\n",
                     i, status, ax, ay, az, gx, gy, gz);
      
      if (ax|ay|az|gx|gy|gz) anyNonZero = true;
    } else {
      Serial.printf("[%2d] Read failed\n", i);
    }
    delay(100);
  }
  
  // ═══ STEP 10: 온도 ═══
  Serial.println("\n--- STEP 10: Temperature ---");
  setBank(0);
  Wire.beginTransmission(ICM_ADDR);
  Wire.write(0x39);
  Wire.endTransmission(false);
  Wire.requestFrom(ICM_ADDR, (uint8_t)2);
  if (Wire.available() >= 2) {
    int16_t t = (Wire.read() << 8) | Wire.read();
    float tempC = (t - 21.0) / 333.87 + 21.0;
    Serial.printf("  Raw=%d → %.1f°C\n", t, tempC);
    if (t != 0) anyNonZero = true;
  }
  
  // ═══ STEP 11: 전체 레지스터 덤프 (디버깅용) ═══
  Serial.println("\n--- STEP 11: Bank 0 Register Dump ---");
  setBank(0);
  for (uint8_t reg = 0x00; reg <= 0x3F; reg += 8) {
    Serial.printf("  0x%02X:", reg);
    for (uint8_t j = 0; j < 8 && (reg+j) <= 0x3F; j++) {
      Serial.printf(" %02X", readReg(reg + j));
    }
    Serial.println();
  }
  
  // ═══ FINAL ═══
  Serial.println("\n============================================");
  if (anyNonZero) {
    Serial.println("  ✅✅✅ 센서 데이터 정상! ✅✅✅");
  } else {
    Serial.println("  ❌ 여전히 0 — 하드웨어 문제 확정");
    Serial.println("  → 이 WCMCU-20948도 불량(가품) 가능성");
    Serial.println("  → 새 ICM20948 모듈 구매 필요");
  }
  Serial.println("============================================\n");
}

void loop() {
  setBank(0);
  
  Wire.beginTransmission(ICM_ADDR);
  Wire.write(0x2D);
  Wire.endTransmission(false);
  Wire.requestFrom(ICM_ADDR, (uint8_t)12);
  
  if (Wire.available() >= 12) {
    int16_t ax = (Wire.read() << 8) | Wire.read();
    int16_t ay = (Wire.read() << 8) | Wire.read();
    int16_t az = (Wire.read() << 8) | Wire.read();
    int16_t gx = (Wire.read() << 8) | Wire.read();
    int16_t gy = (Wire.read() << 8) | Wire.read();
    int16_t gz = (Wire.read() << 8) | Wire.read();
    
    Serial.printf("A:%6d %6d %6d | G:%6d %6d %6d\n",
                   ax, ay, az, gx, gy, gz);
  }
  delay(100);
}
