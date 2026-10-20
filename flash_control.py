"""Reusable BLE helper for the "SyncFlashLED" device -- extracted from
sync_flash.py's interactive CLI so other callers (app.py's capture-start
trigger) can send one command without knowing anything about asyncio/bleak
themselves.

The device's actual sequence timing (color cycle / wait / white flash for
"S", and whatever F/R/G/B/W do) lives entirely in its own firmware -- this
module only knows how to find it over BLE and write one command byte.
"""
import asyncio
import threading

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "SyncFlashLED"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"


async def find_device():
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == DEVICE_NAME:
            return d.address
    raise RuntimeError("Device not found -- make sure it's powered and advertising")


async def send_command_async(command_char, timeout_s=3.0):
    """Discover -> connect -> write one command byte -> wait for (at most)
    one notification -> disconnect. Returns (success, response_text) --
    response_text is None on a timeout (the device may not always notify
    back for every command; that's not itself a failure of the send).
    """
    address = await find_device()
    response = {"text": None}
    event = asyncio.Event()

    def _on_notify(_sender, data):
        response["text"] = data.decode(errors="ignore").strip()
        event.set()

    async with BleakClient(address) as client:
        await client.start_notify(CHARACTERISTIC_UUID, _on_notify)
        try:
            await client.write_gatt_char(CHARACTERISTIC_UUID, command_char[0].encode())
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                pass
        finally:
            await client.stop_notify(CHARACTERISTIC_UUID)
    return True, response["text"]


def send_command(command_char, timeout_s=3.0):
    """Synchronous wrapper around send_command_async -- connects fresh and
    disconnects every call, so BLE scan+connect latency (commonly several
    seconds) is paid every time. Fine for sync_flash.py's interactive CLI
    (a human typing commands one at a time), but too slow for anything
    timing-sensitive -- see FlashConnection below for that case. Never
    raises: any failure comes back as (False, error_message).
    """
    try:
        return asyncio.run(send_command_async(command_char, timeout_s=timeout_s))
    except Exception as exc:
        return False, str(exc)


class FlashConnection:
    """Holds ONE BLE connection to the SyncFlashLED device open for as long
    as this object exists (meant to be created once, at app startup, and
    kept for the app's whole lifetime) -- so triggering a command when a
    recording starts is a single GATT write on an already-open connection,
    not a fresh multi-second scan+connect right at the moment low latency
    matters most. Runs its own dedicated thread with its own asyncio event
    loop (bleak/BLE objects aren't meant to be driven from multiple threads
    or loops), auto-reconnecting with a fixed retry delay if the device
    isn't found yet or drops mid-session.
    """

    def __init__(self, retry_delay_s=5.0):
        self._retry_delay_s = retry_delay_s
        self._loop = None
        self._client = None
        self._connected = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._maintain_connection())

    async def _maintain_connection(self):
        while True:
            try:
                address = await find_device()
                async with BleakClient(address) as client:
                    self._client = client
                    self._connected.set()
                    print(f"[Flash] Connected to {DEVICE_NAME} at {address}")
                    while client.is_connected:
                        await asyncio.sleep(1.0)
                    print("[Flash] Disconnected -- will retry")
            except Exception as exc:
                print(f"[Flash] Connection attempt failed ({exc}) -- retrying in {self._retry_delay_s:.0f}s")
            self._connected.clear()
            self._client = None
            await asyncio.sleep(self._retry_delay_s)

    @property
    def is_connected(self):
        return self._connected.is_set()

    def send_command_nowait(self, command_char):
        """Schedules the write on the connection's own event loop and
        returns IMMEDIATELY without waiting for it to complete -- the whole
        point of a persistent connection is paying zero latency in the
        caller's thread right when a take starts. Success/failure is
        logged asynchronously via a callback once the write actually
        finishes. Returns True if a send was scheduled (connected), False
        if there's currently no live connection to use (logged, not
        raised) -- e.g. the device hasn't been found yet or is powered off.
        """
        if not self._connected.is_set() or self._client is None:
            print(f"[Flash] Not connected -- cannot send '{command_char}'")
            return False

        def _on_done(fut):
            exc = fut.exception()
            if exc is not None:
                print(f"[Flash] Send '{command_char}' failed: {exc}")
            else:
                print(f"[Flash] Sent '{command_char}'")

        future = asyncio.run_coroutine_threadsafe(
            self._client.write_gatt_char(CHARACTERISTIC_UUID, command_char[0].encode()), self._loop,
        )
        future.add_done_callback(_on_done)
        return True
