import serial
import time
import sys

SERIAL_PORT = 'COM12' # !!! IMPORTANT: Confirm this
BAUD_RATE = 115200    # !!! IMPORTANT: Confirm this from radar docs

def simple_serial_receiver():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) # Set a longer timeout
        print(f"Successfully opened serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
        print("Listening for raw data. Press Ctrl+C to stop...")

        bytes_received_count = 0
        last_print_time = time.time()

        while True:
            # Read all available bytes from the input buffer
            # ser.in_waiting gives the number of bytes currently in the buffer
            data = ser.read(ser.in_waiting)
            if data:
                bytes_received_count += len(data)
                # Print raw hex data for debugging
                print(f"Received {len(data)} bytes: {data.hex()}")

            # Periodically print a summary if no data is constantly streaming
            if time.time() - last_print_time > 2: # Every 2 seconds
                if bytes_received_count == 0:
                    print(f"Still waiting for data on {SERIAL_PORT}...")
                bytes_received_count = 0
                last_print_time = time.time()

            time.sleep(0.01) # Small delay to avoid busy-waiting

    except serial.SerialException as e:
        print(f"\nERROR: Serial port error: {e}")
        print("Possible issues:")
        print(f" - Port '{SERIAL_PORT}' does not exist or is already in use.")
        print(" - Baud rate mismatch.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    simple_serial_receiver()