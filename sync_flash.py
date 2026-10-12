import asyncio
from bleak import BleakClient, BleakScanner
import time
import csv
from datetime import datetime

CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

async def find_device():
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == "SyncFlashLED":
            return d.address
    raise RuntimeError("Device not found — make sure it's powered and advertising")

async def interactive_session():
    address = await find_device()
    print(f"Found device at {address}")

    log_file = open("flash_log_ble.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["command", "t_sent", "t_notified", "response"])

    # holds the most recent notification text, and an event to signal it arrived
    last_response = {"text": None}
    response_event = asyncio.Event()

    def notification_handler(sender, data):
        text = data.decode(errors="ignore").strip()
        last_response["text"] = text
        print(f"  Arduino: {text}")
        response_event.set()

    async with BleakClient(address) as client:
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Connected. Type a command (F, S, R, G, B, W) and press Enter.")
        print("Type 'q' or 'quit' to exit.\n")

        loop = asyncio.get_event_loop()

        try:
            while True:
                cmd = await loop.run_in_executor(None, input, "> ")
                cmd = cmd.strip()
                if cmd.lower() in ("q", "quit", "exit"):
                    break
                if not cmd:
                    continue

                response_event.clear()
                last_response["text"] = None

                t_sent = time.time()
                await client.write_gatt_char(CHARACTERISTIC_UUID, cmd[0].encode())

                try:
                    await asyncio.wait_for(response_event.wait(), timeout=3.0)
                    t_notified = time.time()
                except asyncio.TimeoutError:
                    t_notified = None
                    print("  (no notification received within 3s)")

                print(f"[{datetime.now().isoformat()}] '{cmd[0]}' sent @ {t_sent:.3f}"
                      + (f", notified @ {t_notified:.3f}" if t_notified else ""))

                writer.writerow([cmd[0], t_sent, t_notified, last_response["text"]])

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            await client.stop_notify(CHARACTERISTIC_UUID)

    log_file.close()
    print("Connection closed.")

if __name__ == "__main__":
    asyncio.run(interactive_session())