import serial
import time

try:
    print("Opening COM19 at 921600...")
    s = serial.Serial('COM19', 921600, timeout=0.1)
    
    # physical reset
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.05)
    s.setDTR(True)
    s.setRTS(False)
    time.sleep(0.1)
    
    print("Reading boot logs for 3 seconds...")
    start_time = time.time()
    
    with open("firmware_boot.txt", "wb") as f:
        while time.time() - start_time < 3.0:
            if s.in_waiting:
                data = s.read(s.in_waiting)
                f.write(data)
    print("Log saved to firmware_boot.txt")
except Exception as e:
    print(f"Error: {e}")
