
import subprocess
import time
import requests
import os
import base64
import signal

import sys

def test_server():
    print("Starting server...")
    # Start server in a new process group to avoid being killed by window-CLOSE
    server_proc = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=open("server_test_stdout.log", "w"),
        stderr=open("server_test_stderr.log", "w"),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
    
    try:
        # Wait for server to be ready
        print("Waiting for server to be ready...")
        ready = False
        for i in range(30):
            try:
                resp = requests.get("http://localhost:7861/api/telemetry", timeout=2)
                if resp.status_code == 200:
                    ready = True
                    print("Server is ready!")
                    break
            except:
                time.sleep(2)
        
        if not ready:
            print("Server failed to start in time.")
            return
        
        # Send generation request
        print("Sending generation request...")
        payload = {
            "prompt": "a high quality portrait of a cat",
            "width": 512,
            "height": 512,
            "steps": 4,
            "cfg_scale": 1.0,
            "sampler": "euler",
            "scheduler": "ays",
            "model_path": "./include/diffusion_model/flux-2-klein-4b.safetensors"
        }
        
        try:
            # High timeout for generation
            resp = requests.post("http://localhost:7861/api/generate", json=payload, timeout=600)
            if resp.status_code == 200:
                data = resp.json()
                if "image" in data:
                    img_str = data["image"]
                    if "," in img_str:
                        img_str = img_str.split(",")[1]
                    try:
                        with open("test_flux_result.png", "wb") as f:
                            f.write(base64.b64decode(img_str))
                        print("Success! Image saved to test_flux_result.png")
                    except Exception as b64e:
                        print(f"Base64 decode failed: {b64e}")
                        print(f"Image string prefix: {img_str[:100]}...")
                        print(f"Image string suffix: {img_str[-100:]}...")
                        print(f"Image string length: {len(img_str)}")
                else:
                    print(f"Error: No image in response: {data}")
            else:
                print(f"Error: Server returned status code {resp.status_code}")
                print(resp.text)
        except Exception as e:
            print(f"Request failed: {e}")
            
    finally:
        print("Shutting down server...")
        # Send CTRL_BREAK_EVENT to the process group
        os.kill(server_proc.pid, signal.CTRL_BREAK_EVENT)
        server_proc.wait(timeout=10)

if __name__ == "__main__":
    test_server()
