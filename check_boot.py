import serial
import time

try:
    print("Opening COM19...")
    s = serial.Serial('COM19', 921600, timeout=1)
    
    # Wait for any DTR/RTS reset to settle
    time.sleep(2)
    s.reset_input_buffer()
    
    # Send custom reboot command implemented in firmware
    print("Sending REBOOT command...")
    s.write(b"REBOOT\n")
    s.flush()
    
    print("Reading log for 4 seconds...")
    start_time = time.time()
    with open("boot_log.txt", "w", encoding='utf-8') as f:
        while time.time() - start_time < 4:
            if s.in_waiting:
                raw_bytes = s.readline()
                line = raw_bytes.decode('utf-8', errors='replace').strip()
                if line:
                    f.write(line + "\n")
            else:
                time.sleep(0.01)
    
    s.close()
    print("Finished reading.")
except Exception as e:
    print(f"Error: {e}")
