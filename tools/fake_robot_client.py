# tools/fake_robot_client.py
"""Simulates the Haro robot's side of the WebSocket protocol, for manually
testing this server without physical robot hardware. Sends a short burst of
silent PCM audio (not real speech -- useful for confirming the pipeline runs
end-to-end and produces *some* transcript/reply/audio, not for judging STT
quality) then end_of_speech, and prints what comes back.
"""
import asyncio
import json
import sys

import websockets

SAMPLE_RATE = 16000
FRAME_BYTES = 2560  # matches the robot's own outgoing frame size


async def main(url: str) -> None:
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "hello", "session_id": "fake-robot-1"}))

        silence_frame = b"\x00" * FRAME_BYTES
        for _ in range(20):  # ~3.2s of silence
            await ws.send(silence_frame)

        await ws.send(json.dumps({"type": "end_of_speech"}))
        print("sent end_of_speech, waiting for response...")

        audio_bytes_received = 0
        async for message in ws:
            if isinstance(message, bytes):
                audio_bytes_received += len(message)
            else:
                print("received:", message)
                if json.loads(message).get("type") == "response_end":
                    break

        print(f"total audio bytes received: {audio_bytes_received}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765/"
    asyncio.run(main(url))
