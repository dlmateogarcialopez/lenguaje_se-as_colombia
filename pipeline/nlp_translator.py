"""
nlp_translator.py
Spanish Text → LSC Gloss Sequence Translator

Role: Takes a Spanish text input and returns an ordered list of LSC glosses 
that correspond to that text. Uses a rule-based approach + vocabulary lookup.
"""
import re
import json
import logging
import os
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VOCAB_PATH = "D:/LSC/pipeline_output/gloss_index/vocabulary.json"


# Spanish stopwords that often have no direct sign
STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "a", "en", "con", "por", "para", "que", "y",
    "o", "pero", "se", "es", "son", "fue", "ser", "estar",
}

# Spanish→LSC Gloss direct mapping (curated from available vocabulary)
SPANISH_TO_GLOSS = {
    # Saludos
    "hola": "HOLA",
    "buenos dias": "BUENAS",
    "buenas": "BUENAS",
    "buenas noches": "BUENAS_NOCHES",
    "buenas tardes": "BUENAS",
    "adiós": "ADIÓS",
    "adios": "HOLA",  # fallback
    
    # Emociones y estado
    "bien": "BIEN",
    "mal": "MAL",
    "feliz": "FELIZ",
    "enfermo": "ENFERMO",
    "salud": "SALUD",
    
    # Acciones comunes
    "ayuda": "AYUDA",
    "ayudar": "AYUDA",
    "necesitar": "NECESITAR",
    "necesito": "NECESITAR",
    "hablar": "HABLAR",
    "llamar": "LLAMAR",
    "caminar": "CAMINAR",
    "correr": "CORRER",
    "comer": "COMER",
    "dormir": "DORMIR",
    "jugar": "JUGAR",
    "vivir": "VIVIR",
    "trabajar": "TRABAJO",
    
    # Personas y relaciones
    "amigo": "AMIGO",
    "familia": "FAMILIA",
    "madre": "MADRE",
    "padre": "PADRE",
    "niño": "NIÑO",
    "hombre": "HOMBRE",
    "mujer": "MUJER",
    "estudiante": "ESTUDIANTE",
    "policia": "POLICIA",
    
    # Lugares y cosas
    "casa": "CASA",
    "hospital": "HOSPITAL",
    "pan": "PAN",
    "leche": "LECHE",
    "cafe": "CAFE",
    "dinero": "DINERO",
    
    # Descriptores
    "grande": "GRANDE",
    "pequeño": "PEQUEÑO",
    "amor": "AMOR",
    "comunicacion": "COMUNICACION",
    
    # Tiempo
    "mañana": "MAÑANA",
    "noche": "NOCHE",
    "tarde": "TARDE",
    "mes": "MES",
    "lunes": "LUNES",
    "miércoles": "MIERCOLES",
    "miercoles": "MIERCOLES",
    "viernes": "VIERNES",
    "sábado": "SABADO",
    "sabado": "SABADO",
    
    # Cortesia
    "gracias": "GRACIAS",
    "perdón": "PERDON",
    "perdon": "PERDON",
    "pregunta": "PREGUNTA",
    "numero": "NUMERO",
}

def load_vocabulary() -> List[str]:
    """Load available glosses from index."""
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            return json.load(f)
    # Fallback: return keys from mapping
    return list(set(SPANISH_TO_GLOSS.values()))

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[¿?¡!.,;:"\']', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def translate_to_glosses(text: str, vocabulary: List[str] = None) -> List[str]:
    """
    Translates Spanish text to an ordered list of LSC glosses.
    
    Strategy:
    1. Check multi-word phrases first (e.g. "buenos días")
    2. Check individual words
    3. If word not in dict, check if its stem is in vocab
    4. Skip stopwords with no sign equivalent
    """
    if vocabulary is None:
        vocabulary = load_vocabulary()
    
    vocab_set = {v.upper() for v in vocabulary}    
    normalized = normalize_text(text)
    tokens = normalized.split()
    
    glosses = []
    i = 0
    
    while i < len(tokens):
        # Try 3-grams, 2-grams, then 1-gram
        matched = False
        for n in [3, 2, 1]:
            phrase = " ".join(tokens[i:i+n])
            gloss = SPANISH_TO_GLOSS.get(phrase)
            if gloss and (gloss in vocab_set or len(vocab_set) == 0):
                glosses.append(gloss)
                i += n
                matched = True
                break
        
        if not matched:
            word = tokens[i]
            if word not in STOPWORDS:
                # Try uppercase match against known vocab
                upper = word.upper()
                if upper in vocab_set:
                    glosses.append(upper)
                else:
                    # Word not available — mark as unknown
                    logging.warning(f"No gloss found for: '{word}' — skipping")
            i += 1
    
    logging.info(f"Input: '{text}' → Glosses: {glosses}")
    return glosses


if __name__ == "__main__":
    vocab = load_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test translations
    tests = [
        "Hola, ¿cómo estás?",
        "Necesito ayuda, estoy enfermo",
        "Gracias amigo",
        "Buenos días, mi familia está bien",
        "El niño camina a la casa",
    ]
    
    print("\n--- Translation Tests ---")
    for t in tests:
        glosses = translate_to_glosses(t, vocab)
        print(f"  '{t}' → {glosses}")
