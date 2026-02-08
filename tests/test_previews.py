
import asyncio
import json
import websockets
import requests
import time
import subprocess
import sys
import os

async def test_preview():
    # Start the server
    server_process = subprocess.Popen([sys.executable, "server.py"])
    print("Waiting for server to start...")
    time.sleep(15) # Wait for server to init

    uri = "ws://localhost:7861/ws/preview"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            # Start generation in a separate thread/task
            def start_gen():
                payload = {
                    "prompt": "a beautiful landscape",
                    "width": 512,
                    "height": 512,
                    "steps": 10,
                    "enable_preview": True
                }
                response = requests.post("http://localhost:7861/api/generate", json=payload)
                print("Generation requested:", response.status_code)

            import threading
            threading.Thread(target=start_gen).start()

            # Listen for messages
            previews_received = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    print(f"Received message type: {data.get('type')} step: {data.get('step')}")
                    if data.get("type") == "preview":
                        previews_received += 1
                        if "images" in data and len(data["images"]) > 0:
                            print(f"Preview contains {len(data['images'])} images")
                            # Save the first image to disk
                            if previews_received == 1:
                                import base64
                                img_data = data["images"][0].split(",")[1]
                                with open("test_preview_0.jpg", "wb") as f:
                                    f.write(base64.b64decode(img_data))
                                print("Saved test_preview_0.jpg")
                        else:
                            print("Preview MISSING images!")
                except asyncio.TimeoutError:
                    print("Timeout waiting for message")
                except Exception as e:
                    print(f"Error: {e}")
                    break
            
            print(f"Total previews received: {previews_received}")
            if previews_received > 0:
                print("SUCCESS: Received previews via WebSocket")
            else:
                print("FAILURE: No previews received")

    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    asyncio.run(test_preview())
