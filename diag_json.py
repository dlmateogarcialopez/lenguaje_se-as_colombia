import json
with open('d:/LSC/webapp/lsc_motion_dummy.json') as f:
    data = json.load(f)

if isinstance(data, dict):
    frames = data.get('frames', [])
    print(f"Label: {data.get('label', 'Unknown')}")
else:
    frames = data
    print("Format: List (Raw Frames)")

print(f"Num Frames: {len(frames)}")

if frames:
    f0 = frames[0]
    for key in ['poseLandmarks', 'leftHandLandmarks', 'rightHandLandmarks', 'faceLandmarks']:
        lms = f0.get(key, [])
        if lms:
            xs = [p['x'] for p in lms]
            ys = [p['y'] for p in lms]
            zs = [p['z'] for p in lms]
            print(f"{key} range - X: [{min(xs):.4f}, {max(xs):.4f}], Y: [{min(ys):.4f}, {max(ys):.4f}], Z: [{min(zs):.4f}, {max(zs):.4f}]")
        else:
            print(f"{key} is empty")
