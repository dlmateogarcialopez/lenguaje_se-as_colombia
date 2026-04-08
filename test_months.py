
import time
import sys
import logging
from pipeline.nlp_translator import translate_to_glosses
from pipeline.motion_synthesizer import synthesize_sequence

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

phrase = '31 de Octubre'
print(f"--- TESTING LSC TRANSLATION: {phrase} ---")

try:
    # 1. Translate
    glosses = translate_to_glosses(phrase)
    print(f"Glosses: {glosses}")

    # 2. Synthesize
    start = time.time()
    result = synthesize_sequence(glosses)
    end = time.time()

    print(f"--- SUCCESS ---")
    print(f"Synthesized {len(result['frames'])} frames in {end-start:.2f}s")
    
    # Check if the last sign (Octubre) is present
    last_frames = [f for f in result['frames'] if f['gloss'] == 'MES_OCTUBRE']
    print(f"Frames for MES_OCTUBRE: {len(last_frames)}")
    
    if len(last_frames) > 0:
        print(f"Sample landmark (pose[0]): {last_frames[0]['poseLandmarks'][0]}")
    
except Exception as e:
    print(f"--- ERROR ---")
    print(f"Error during synthesis: {e}")
    import traceback
    traceback.print_exc()
