import pandas as pd
df = pd.read_csv(r"d:\LSC\pipeline_output\master_normalized.csv", nrows=1)
cols = df.columns.tolist()
meta_cols = ['source', 'signer', 'topic', 'sign_label', 'video_id', 'repetition', 'frame_id']
feat_cols = [c for c in cols if c not in meta_cols]

print(f"Total features: {len(feat_cols)}")
for c in feat_cols[:10]:
    print(c)
print("...")
for c in feat_cols[-10:]:
    print(c)

print("Unique prefixes:")
prefixes = set([c.rsplit('_', 2)[0] if '_' in c else c for c in feat_cols])
print(prefixes)
