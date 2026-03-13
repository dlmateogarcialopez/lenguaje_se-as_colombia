import os
import shutil

LSC50_BODY_DIR = "D:/LSC/LSC50/LANDMARKS/BODY_LANDMARKS"
DYNAMIC_DIR = "D:/LSC/pipeline_output/dynamic_landmarks"
os.makedirs(DYNAMIC_DIR, exist_ok=True)

# We will use "NUMERO" (0037) as base for all numbers for now
# and "AYUDA" (0002) as base for all letters for now.
# This is just TO VALIDATE THE PIPELINE WORKS.

def generate_mocks():
    # Find a sample for NUMERO
    numero_sample = os.path.join(LSC50_BODY_DIR, "0037_0000_0000.csv")
    ayuda_sample = os.path.join(LSC50_BODY_DIR, "0002_0000_0000.csv")
    
    if not os.path.exists(numero_sample):
        # Fallback to any file in that dir
        any_file = os.listdir(LSC50_BODY_DIR)[0]
        numero_sample = os.path.join(LSC50_BODY_DIR, any_file)
        ayuda_sample = numero_sample

    # Digits 0-9
    for i in range(10):
        dst = os.path.join(DYNAMIC_DIR, f"NUMERO_{i}.csv")
        shutil.copy(numero_sample, dst)
        
    # Letters A-Z
    import string
    for char in string.ascii_uppercase:
        dst = os.path.join(DYNAMIC_DIR, f"LETRA_{char}.csv")
        shutil.copy(ayuda_sample, dst)
    
    # Special NN
    shutil.copy(ayuda_sample, os.path.join(DYNAMIC_DIR, "LETRA_NN.csv"))

    print(f"Mock dynamic landmarks generated in {DYNAMIC_DIR}")

if __name__ == "__main__":
    generate_mocks()
