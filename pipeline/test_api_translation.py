import requests
import json

def test():
    phrase = "Se facturaron 31 días que se componen de"
    url = "http://localhost:8000/translate"
    
    print(f"Testing API with phrase: '{phrase}'")
    try:
        r = requests.post(url, json={"text": phrase}, timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"SUCCESS!")
        print(f"Glosses: {data['glosses']}")
        print(f"Total frames: {data['total_frames']}")
        
        # Test unknown word (fingerspelling)
        phrase2 = "Municipio Circasia"
        print(f"\nTesting unknown word: '{phrase2}'")
        r2 = requests.post(url, json={"text": phrase2}, timeout=10)
        print(f"Glosses: {r2.json()['glosses']}")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test()
