import pandas as pd
df = pd.read_csv(r"d:\LSC\pipeline_output\master_normalized.csv", nrows=1)

ordered_features = []
for i in range(33): ordered_features.extend([f"pose_{i}_x", f"pose_{i}_y", f"pose_{i}_z"])
for i in range(21): ordered_features.extend([f"l_hand_{i}_x", f"l_hand_{i}_y", f"l_hand_{i}_z"])
for i in range(21): ordered_features.extend([f"r_hand_{i}_x", f"r_hand_{i}_y", f"r_hand_{i}_z"])
for i in range(468): ordered_features.extend([f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"])

missing = [c for c in ordered_features if c not in df.columns]
print(f"Total exactly required: {len(ordered_features)}")
print(f"Missing from df: {len(missing)}")
if missing:
    print("Example missing:", missing[:10])
