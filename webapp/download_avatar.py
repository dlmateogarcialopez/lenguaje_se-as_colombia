import urllib.request
import os

url = "https://models.readyplayer.me/63b7e70878a1eece35b1c4b6.glb"
out_path = "d:/LSC/webapp/rpm_avatar.glb"

try:
    print("Downloading realistic RPM Avatar...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Saved to {out_path} ({os.path.getsize(out_path)} bytes)")
except Exception as e:
    print(f"Download failed: {e}")
