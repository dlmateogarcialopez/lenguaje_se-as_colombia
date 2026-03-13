import sys
import os

# Ensure modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp_translator import translate_to_glosses, load_vocabulary
from motion_synthesizer import synthesize_sequence, export_to_json

def verify():
    # Example phrase from the dashboard
    phrases = [
        "Se facturaron 31 días que se componen de",
        "Días de consumo total 31",
        "Municipio Circasia",
        "Servicio de Energía"
    ]
    
    vocab = load_vocabulary()
    
    print("--- ADVANCED LSC TRANSLATION VERIFICATION ---\n")
    
    for phrase in phrases:
        print(f"SPANISH: '{phrase}'")
        # 1. Translate to Glosses
        glosses = translate_to_glosses(phrase, vocab)
        print(f"GLOSSES: {glosses}")
        
        # 2. Synthesize to Motion JSON
        result = synthesize_sequence(glosses)
        print(f"MOTION: {result['total_frames']} frames generated.")
        
        # 3. Export
        out = export_to_json(result, f"verify_{phrase.replace(' ', '_')[:20]}.json")
        print(f"EXPORTED: {out}\n")

if __name__ == "__main__":
    verify()
