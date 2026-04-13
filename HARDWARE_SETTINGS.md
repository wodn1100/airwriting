# AirWriting 하드웨어 & 통신 설정 레퍼런스

## 센서 구성

| ID | 역할 | 센서 | I2C Bus | I2C 주소 | 비고 |
|----|------|------|---------|----------|------|
| S1 | 팔뚝 (forearm) | MPU6050 | Bus 0 | `0x68` (104) | AD0=LOW |
| S2 | 손 (hand) | MPU6050 | Bus 0 | `0x69` (105) | AD0=HIGH |
| S3 | 손가락 (finger) | ICM20948 | Bus 1 | `0x68` (104) | 별도 버스 |
| Mag | 지자기 (S3 내장) | AK09916 | Bus 1 | `0x0C` (12) | ICM20948 내장 |
| OLED | 디스플레이 | SSD1306 | Bus 0 | `0x3C` (60) | 128×64 |

## ESP32-S3 핀 배치

### I2C
| 버스 | SDA | SCL | 속도 |
|------|-----|-----|------|
| Bus 0 (MPU×2 + OLED) | GPIO 21 | GPIO 22 | 400kHz |
| Bus 1 (ICM20948) | GPIO 32 | GPIO 33 | 400kHz |

### GPIO
| 핀 | 용도 |
|-----|------|
| GPIO 15 | 펜 버튼 (INPUT_PULLUP) |

## UDP 통신 포트

| 포트 | 용도 |
|------|------|
| 12344 | Discovery (서버 자동 탐색) |
| 12345 | ESP → PC/Jetson (IMU 데이터 수신) |
| 12346 | PC/Jetson → Unity/Web (시각화 전송) |
| 12348 | Action Dispatch / Web UI |
| 12349 | Phone (모바일 앱) |
| 12350 | Python Control |
| 5555 | ESP32 로컬 UDP 포트 |

## 센서 스케일 설정

| 센서 | Accel Range | Accel Scale | Gyro Range | Gyro Scale |
|------|-------------|-------------|------------|------------|
| MPU6050 | ±8g | `9.81 / 4096.0` m/s² | ±1000°/s | `(π/180) / 32.8` rad/s |
| ICM20948 | ±8g | `9.81 / 4096.0` m/s² | ±1000°/s | `(π/180) / 32.8` rad/s |
| AK09916 | — | — | — | `0.15` µT/LSB |

## 패킷 구조 (V3)

```
[Header 0xAA] [Timestamp 4B] [S1 24B] [S2 24B] [S3 36B] [Button 1B] [Checksum 1B] [Footer 0x55]

S1/S2 (SensorData6): ax,ay,az,gx,gy,gz (각 float32 = 24바이트)
S3 (SensorData9): ax,ay,az,gx,gy,gz,mx,my,mz (각 float32 = 36바이트)
```

## 기타

- **Serial Baud Rate**: 115200
- **샘플링 주파수**: 100Hz
- **WiFi 설정**: WiFiManager 캡티브 포털 (`AirWriting_Setup`)
- **서버 탐색 순서**: mDNS → UDP Broadcast → Fallback IP
- **mDNS 호스트**: `airwriting-glove`
