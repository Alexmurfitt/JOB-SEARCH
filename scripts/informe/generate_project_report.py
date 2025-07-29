#!/usr/bin/env python3
"""
generate_project_report.py

Genera un informe en Markdown de la estructura y contenido del proyecto,
respetando un límite práctico de tamaño de informe.
"""

import os
from pathlib import Path

# Configuración
ROOT = Path(__file__).parent.resolve()
OUTPUT = ROOT / "estructura_resumida.md"
MAX_CONTENT_BYTES = 10 * 1024      # Incluir contenido completo hasta 10 KB
MAX_PREVIEW_LINES = 50             # Líneas a mostrar si el archivo supera MAX_CONTENT_BYTES
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv"}  # directorios a ignorar

def human_size(n):
    """Convierte bytes a una cadena más legible."""
    for unit in ("B","KB","MB","GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def is_text_file(path: Path):
    """Prueba rápida: ext de texto común."""
    return path.suffix.lower() in {
        ".py", ".md", ".yaml", ".yml", ".txt", ".json", ".csv", ".ini", ".cfg", ".xml", ".html"
    }

def main():
    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write("# Informe de la estructura del proyecto\n\n")
        for dirpath, dirnames, filenames in os.walk(ROOT):
            # Saltar entornos y .git
            rel_dir = Path(dirpath).relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel_dir.parts):
                continue

            # Escribir cabecera de directorio
            depth = len(rel_dir.parts)
            out.write(f"{'#'*(depth+1)} {rel_dir or 'Raíz'}\n\n")

            for fn in sorted(filenames):
                path = Path(dirpath) / fn
                # Omitir el propio informe
                if path == OUTPUT:
                    continue

                size = path.stat().st_size
                out.write(f"- **{rel_dir / fn}** ({human_size(size)})\n")

                # Si es archivo de texto pequeño, incluir completo
                if is_text_file(path) and size <= MAX_CONTENT_BYTES:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    out.write("  ```\n")
                    out.write(text.rstrip() + "\n")
                    out.write("  ```\n")
                # Si es texto grande, incluir sólo un preview
                elif is_text_file(path):
                    out.write("  ```\n")
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= MAX_PREVIEW_LINES:
                                out.write(f"... _solo primeras {MAX_PREVIEW_LINES} líneas_\n")
                                break
                            out.write(line.rstrip() + "\n")
                    out.write("  ```\n")

                # Para binarios o extensiones no listadas, no muestro contenido
            out.write("\n")

    print(f"✔ Informe generado: {OUTPUT}")

if __name__ == "__main__":
    main()
