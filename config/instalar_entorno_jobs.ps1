# instalar_entorno_jobs.ps1
# Este script instala todas las dependencias necesarias para el proyecto de análisis de formación y salario

Write-Host "🔁 Activando entorno virtual venv_jobs..."
& ".\venv_jobs\Scripts\Activate.ps1"

Write-Host "📦 Instalando librerías desde requirements.txt (versión corregida)..."
# Crea temporalmente un requirements actualizado
@'
# NLP y NER
transformers==4.51.3
tokenizers==0.21.1

# ML y DL
scikit-learn==1.4.2
xgboost==3.0.0
lightgbm==4.6.0
catboost==1.2.8
shap==0.47.2
joblib==1.5.0

# Preprocesamiento
pandas==2.2.2
numpy==1.26.4

# Visualización
matplotlib==3.9.0
plotly==5.21.0
seaborn==0.13.2

# Reportes PDF/HTML
fpdf==1.7.2
reportlab==4.1.0
nbconvert==7.16.3

# API y despliegue
fastapi==0.110.2
uvicorn==0.29.0
docker==7.0.0

# Notebooks y Jupyter
jupyterlab==4.1.5
notebook==7.0.6
ipykernel==6.29.4
'@ | Out-File -Encoding UTF8 -FilePath "requirements_actualizado.txt"

pip install -r requirements_actualizado.txt

Write-Host "`n✅ Entorno configurado con éxito. Puedes ejecutar tu script con: python generador_ruta_formativa.py"
