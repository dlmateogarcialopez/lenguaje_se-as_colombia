import numpy as np
import pickle
y = np.load('d:/LSC/pipeline_output/LSC_labels.npy', allow_pickle=True)
with open('d:/LSC/pipeline_output/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
label_idx = np.argmax(y[0])
print(le.inverse_transform([label_idx])[0])
