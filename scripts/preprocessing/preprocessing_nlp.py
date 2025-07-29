import re
from langdetect import detect
from deep_translator import GoogleTranslator


def clean_text(text):
    """Limpieza general del texto."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)  # eliminar HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)  # eliminar URLs
    text = re.sub(r'[^\w\s.,]', '', text)  # eliminar símbolos
    text = re.sub(r'\s+', ' ', text)  # espacios múltiples
    return text.strip()


def detect_language(text):
    """Detecta el idioma del texto."""
    try:
        return detect(text)
    except:
        return "unknown"


def translate_to_english(text):
    """Traduce el texto al inglés si no está en inglés."""
    lang = detect_language(text)
    if lang != 'en' and lang != 'unknown':
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        except:
            return text
    return text


def preprocess_job_posting(text):
    """Pipeline de preprocesamiento completo para una oferta de empleo."""
    text = clean_text(text)
    text = translate_to_english(text)
    return text


# Ejemplo de uso
if __name__ == "__main__":
    raw_text = "<div>Se busca Data Analyst con experiencia en Python, SQL y Power BI. Salario: 50.000€</div>"
    processed = preprocess_job_posting(raw_text)
    print(processed)
