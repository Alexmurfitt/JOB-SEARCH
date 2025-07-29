import subprocess
import os
import time

def ejecutar(nombre_script):
    print(f"\n🚀 Ejecutando: {nombre_script}")
    resultado = subprocess.run(["python", nombre_script], capture_output=True, text=True)
    print(resultado.stdout)
    if resultado.stderr:
        print("⚠️ Error:", resultado.stderr)

def main():
    # 1. Pipeline completo de preprocesamiento
    ejecutar("scripts/main_pipeline_jobs.py")

    # 2. Modelado predictivo y SHAP
    ejecutar("scripts/analysis_and_modeling_advanced.py")

    # 3. Informe estructurado ML
    ejecutar("scripts/generar_informe.py")

    # 4. Informe semántico textual con spaCy
    ejecutar("scripts/generar_informe_data_analyst.py")

    print("\n✅ Sistema completo ejecutado correctamente.")

    # 5. Opción para abrir el dashboard
    respuesta = input("\n¿Deseas abrir el dashboard interactivo ahora? (s/n): ").strip().lower()
    if respuesta == "s":
        subprocess.run(["streamlit", "run", "scripts/dashboard.py"])

if __name__ == "__main__":
    main()

