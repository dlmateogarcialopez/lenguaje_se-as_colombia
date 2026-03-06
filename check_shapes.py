import numpy as np
try:
    X = np.load(r'd:\LSC\pipeline_output\LSC_dataset_unificado.npy')
    y = np.load(r'd:\LSC\pipeline_output\LSC_labels.npy')
    print('X shape:', X.shape)
    print('y shape:', y.shape)
except Exception as e:
    print('Error:', e)
