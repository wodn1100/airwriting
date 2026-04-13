import serial
import time
import re

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
    raw_buffer = b""
    while time.time() - start_time < 3.0:
        if s.in_waiting:
            raw_buffer += s.read(s.in_waiting)
            
    print("Log size:", len(raw_buffer))
    # Find ascii strings
    strings = re.findall(b'[a-zA-Z0-9_\-\.\=\[\]\:\! ]{5,}', raw_buffer)
    for string in strings:
        if b"U" not in string: # ignore the binary UUUU
            print(string.decode('ascii'))
            
except Exception as e:
    print(f"Error: {e}")
