import asyncio
import csv
import time
from datetime import datetime

import flash_control


async def interactive_session():
    log_file = open("flash_log_ble.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["command", "t_sent", "t_notified", "response"])

    print("Type a command (F, S, R, G, B, W) and press Enter.")
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

            t_sent = time.time()
            try:
                _success, response_text = await flash_control.send_command_async(cmd[0])
            except RuntimeError as exc:
                print(f"  {exc}")
                continue
            t_notified = time.time() if response_text is not None else None

            if response_text is not None:
                print(f"  Arduino: {response_text}")
            else:
                print("  (no notification received within 3s)")

            print(f"[{datetime.now().isoformat()}] '{cmd[0]}' sent @ {t_sent:.3f}"
                  + (f", notified @ {t_notified:.3f}" if t_notified else ""))

            writer.writerow([cmd[0], t_sent, t_notified, response_text])

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        log_file.close()
        print("Connection closed.")

if __name__ == "__main__":
    asyncio.run(interactive_session())
