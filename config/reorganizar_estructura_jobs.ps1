# Script PowerShell para reorganizar la estructura del proyecto SEARCH_JOB

# Definir ruta base
$base = "D:\1. FORMACION Y EMPLEO\SEARCH_JOB"

# Crear carpetas destino si no existen
$folders = @(
    "scripts\etl",
    "scripts\scraping",
    "scripts\ner",
    "scripts\ml",
    "scripts\dashboard",
    "scripts\informe",
    "scripts\preprocessing",
    "scripts\utils",
    "data\raw",
    "data\processed",
    "data\splits",
    "data\features",
    "data\annotate",
    "data\annotated",
    "data\new_splits",
    "models\ner",
    "models\salary",
    "models\baseline",
    "models\catboost_info",
    "img",
    "docs",
    "notebooks",
    "reports",
    "tests",
    "config"
)

foreach ($folder in $folders) {
    $fullPath = Join-Path $base $folder
    if (!(Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath | Out-Null
    }
}

# Mover archivos a sus carpetas correspondientes (ejemplos clave)
# NOTA: Añadir o ajustar más según sea necesario

# Scripts ETL
Move-Item "$base\scripts\etl.py" "$base\scripts\etl" -Force

# Scripts NER
Get-ChildItem "$base\scripts\*ner*" -File | ForEach-Object {
    Move-Item $_.FullName "$base\scripts\ner" -Force
}

# Scripts de scraping
$patternsScraping = "*scraper*", "*spider*", "*scrapy*", "*linkedin*"
foreach ($pattern in $patternsScraping) {
    Get-ChildItem "$base\scripts\$pattern" -File | ForEach-Object {
        Move-Item $_.FullName "$base\scripts\scraping" -Force
    }
}

# Scripts de ML
$patternsML = "*train_salary*", "*salary_model*", "*modeling*", "*predict_salary*", "*stack*"
foreach ($pattern in $patternsML) {
    Get-ChildItem "$base\scripts\$pattern" -File | ForEach-Object {
        Move-Item $_.FullName "$base\scripts\ml" -Force
    }
}

# Scripts de informes
$patternsInforme = "*generar_informe*", "*generate_project_report*", "*ruta_formativa*"
foreach ($pattern in $patternsInforme) {
    Get-ChildItem "$base\scripts\$pattern" -File | ForEach-Object {
        Move-Item $_.FullName "$base\scripts\informe" -Force
    }
}

# Scripts de preprocesamiento
$patternsPrep = "*preprocess*", "*preprocessing*", "*convertir_salario*"
foreach ($pattern in $patternsPrep) {
    Get-ChildItem "$base\scripts\$pattern" -File | ForEach-Object {
        Move-Item $_.FullName "$base\scripts\preprocessing" -Force
    }
}

# Mover archivos de configuración
Move-Item "$base\*.txt" "$base\config" -Force
Move-Item "$base\*.cfg" "$base\config" -Force
Move-Item "$base\*.ps1" "$base\config" -Force
Move-Item "$base\*.bat" "$base\config" -Force
Move-Item "$base\*.env" "$base\config" -Force