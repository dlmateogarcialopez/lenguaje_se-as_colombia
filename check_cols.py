import pandas as pd

try:
    df_lsc50 = pd.read_csv(r"d:\LSC\pipeline_output\lsc50_interim.csv")
    print("\nLSC50 Sample Columns:")
    print([c for c in df_lsc50.columns if c not in ['source', 'signer', 'topic', 'sign_label', 'video_id', 'repetition', 'frame_id']][:20])
except Exception as e:
    print("Error LSC50:", e)

try:
    df_lsc54 = pd.read_csv(r"d:\LSC\pipeline_output\lsc54_interim.csv")
    print("\nLSC54 Sample Columns:")
    print([c for c in df_lsc54.columns if c not in ['source', 'signer', 'topic', 'sign_label', 'video_id', 'repetition', 'frame_id']][:20])
except Exception as e:
    print("Error LSC54:", e)

try:
    df_lsc70 = pd.read_csv(r"d:\LSC\pipeline_output\lsc70_interim.csv")
    print("\nLSC70 Sample Columns:")
    print([c for c in df_lsc70.columns if c not in ['source', 'signer', 'topic', 'sign_label', 'video_id', 'repetition', 'frame_id']][:20])
    
    print("\nTotal expected feature columns:", len([c for c in df_lsc70.columns if c not in ['source', 'signer', 'topic', 'sign_label', 'video_id', 'repetition', 'frame_id']]))
except Exception as e:
    print("Error LSC70:", e)
