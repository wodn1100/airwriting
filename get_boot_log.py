import serial
import time
import sys

try:
    print("Opening COM19 at 115200...")
    s = serial.Serial('COM19', 115200, timeout=0.1)
    
    # Toggle DTR/RTS to physically reset the ESP32
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.05)
    s.setDTR(True)
    s.setRTS(False)
    time.sleep(0.05)
    
    print("Reading boot logs for 3 seconds...")
    start_time = time.time()
    
    with open("boot_log.txt", "w", encoding="utf-8") as f:
        while time.time() - start_time < 3.0:
            if s.in_waiting:
                # read byte by byte or lines
                lines = s.readlines()
                for raw_line in lines:
                    line = raw_line.decode('utf-8', errors='replace').strip()
                    if line:
                        f.write(line + "\n")
    s.close()
    print("Boot log captured successfully.")
except Exception as e:
    print(f"Error: {e}")
