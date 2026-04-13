"""S1/S2/S3 센서 전부 확인 + 패킷 hex dump"""
import serial, struct, time

s = serial.Serial('COM19', 921600, timeout=3)
time.sleep(2)

PACKET_SIZE = 92
raw = s.read(PACKET_SIZE * 50)
s.close()

print(f"Read {len(raw)} bytes")

# Packet layout:
# [0]    header  0xAA
# [1-4]  timestamp (uint32)
# [5-28]  S1 (6 floats: ax,ay,az,gx,gy,gz)
# [29-52] S2 (6 floats: ax,ay,az,gx,gy,gz)
# [53-88] S3 (9 floats: ax,ay,az,gx,gy,gz,mx,my,mz)
# [89]   button
# [90]   checksum
# [91]   footer 0x55

i = 0
count = 0
while i < len(raw) - PACKET_SIZE:
    if raw[i] == 0xAA and raw[i + PACKET_SIZE - 1] == 0x55:
        pkt = raw[i:i + PACKET_SIZE]

        if count < 3:
            # S1
            s1 = struct.unpack_from('<6f', pkt, 5)
            # S2
            s2 = struct.unpack_from('<6f', pkt, 29)
            # S3
            s3 = struct.unpack_from('<9f', pkt, 53)

            print(f"\n--- Packet #{count} ---")
            print(f"  S1 accel: [{s1[0]:.3f}, {s1[1]:.3f}, {s1[2]:.3f}]")
            print(f"  S1 gyro:  [{s1[3]:.4f}, {s1[4]:.4f}, {s1[5]:.4f}]")
            print(f"  S2 accel: [{s2[0]:.3f}, {s2[1]:.3f}, {s2[2]:.3f}]")
            print(f"  S2 gyro:  [{s2[3]:.4f}, {s2[4]:.4f}, {s2[5]:.4f}]")
            print(f"  S3 accel: [{s3[0]:.3f}, {s3[1]:.3f}, {s3[2]:.3f}]")
            print(f"  S3 gyro:  [{s3[3]:.4f}, {s3[4]:.4f}, {s3[5]:.4f}]")
            print(f"  S3 mag:   [{s3[6]:.2f}, {s3[7]:.2f}, {s3[8]:.2f}]")
            print(f"  Button: {pkt[89]}")

            # Hex dump of S3 region
            print(f"  S3 hex: {pkt[53:89].hex()}")

        count += 1
        i += PACKET_SIZE
    else:
        i += 1

print(f"\nTotal valid packets: {count}")
