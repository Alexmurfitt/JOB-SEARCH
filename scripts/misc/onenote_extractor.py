import os
import json
import requests
import logging
from msal import PublicClientApplication
from bs4 import BeautifulSoup
from datetime import datetime

# Configura los logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# -------------------- CONFIGURACIÓN --------------------
CLIENT_ID = 'TU_CLIENT_ID'
TENANT_ID = 'TU_TENANT_ID'
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ['Notes.Read.All']
OUTPUT_DIR = "OneNote_Export"
# -------------------------------------------------------

# Inicializa MSAL para autenticación
app = PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY)
result = app.acquire_token_interactive(scopes=SCOPES)
token = result.get("access_token")

headers = {
    'Authorization': f'Bearer {token}'
}

# Crea directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Función para limpiar contenido HTML
def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator='\n').strip()

# Función principal de extracción
def extract_onenote():
    all_data = []

    logging.info("Obteniendo blocs de notas...")
    notebooks = requests.get('https://graph.microsoft.com/v1.0/me/onenote/notebooks', headers=headers).json().get('value', [])

    for nb in notebooks:
        nb_name = nb['displayName'].strip().replace("/", "_")
        logging.info(f"📒 Bloc: {nb_name}")
        nb_path = os.path.join(OUTPUT_DIR, nb_name)
        os.makedirs(nb_path, exist_ok=True)

        sections = requests.get(nb['sectionsUrl'], headers=headers).json().get('value', [])
        for sec in sections:
            sec_name = sec['displayName'].strip().replace("/", "_")
            logging.info(f"  📂 Sección: {sec_name}")
            sec_path = os.path.join(nb_path, sec_name)
            os.makedirs(sec_path, exist_ok=True)

            pages_url = f"https://graph.microsoft.com/v1.0/me/onenote/sections/{sec['id']}/pages"
            pages = requests.get(pages_url, headers=headers).json().get('value', [])

            for pg in pages:
                title = pg['title'].strip().replace("/", "_")
                page_id = pg['id']
                page_content_url = pg['contentUrl']
                logging.info(f"    📄 Página: {title}")

                try:
                    html_resp = requests.get(page_content_url, headers=headers)
                    text_content = extract_text_from_html(html_resp.text)

                    filename = os.path.join(sec_path, f"{title}.txt")
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(text_content)

                    all_data.append({
                        "notebook": nb_name,
                        "section": sec_name,
                        "page": title,
                        "content": text_content
                    })

                except Exception as e:
                    logging.warning(f"❌ Error procesando {title}: {e}")
    
    # Exporta CSV y JSON
    export_json = os.path.join(OUTPUT_DIR, "export_onenote.json")
    export_csv = os.path.join(OUTPUT_DIR, "export_onenote.csv")

    with open(export_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    import pandas as pd
    df = pd.DataFrame(all_data)
    df.to_csv(export_csv, index=False, encoding="utf-8")

    logging.info(f"✅ Exportación finalizada: {len(all_data)} páginas procesadas.")
    logging.info(f"📁 Archivos guardados en: {OUTPUT_DIR}")

# Ejecuta todo
if __name__ == "__main__":
    extract_onenote()
