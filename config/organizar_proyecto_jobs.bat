@echo off
REM ─────────────────────────────────────────────
REM ORGANIZADOR AUTOMÁTICO DE SEARCH_JOB
REM Estructura profesional segura y no destructiva
REM Autor: ChatGPT para Alexander
REM ─────────────────────────────────────────────

setlocal EnableDelayedExpansion

REM Ruta base del proyecto
set "BASE_DIR=%~dp0"

REM Directorios estándar
set DIRS=scripts data\models data\raw data\processed models reports dashboards notebooks img docs scrapy_employment_scraper scrapy_employment_scraper\spiders config

echo.
echo 📁 Creando carpetas principales si no existen...
for %%D in (%DIRS%) do (
    if not exist "!BASE_DIR!%%D" (
        mkdir "!BASE_DIR!%%D"
        echo [+] Carpeta creada: %%D
    )
)

echo.
echo 🔁 Reorganizando archivos por extensión...

REM Scripts de Python → scripts/
move *.py scripts\ 2>nul

REM Configuraciones y entorno
move *.env config\ 2>nul
move *.cfg config\ 2>nul
move *.yaml config\ 2>nul
move *.yml config\ 2>nul

REM Modelos
move *.pkl models\ 2>nul
move *.joblib models\ 2>nul
move *.sav models\ 2>nul

REM Documentación
move *.md docs\ 2>nul
move *.pdf docs\ 2>nul
move estructura*.txt docs\ 2>nul

REM Imágenes
move *.png img\ 2>nul
move *.jpg img\ 2>nul
move *.jpeg img\ 2>nul

REM Datos
move *.csv data\raw\ 2>nul
move *.json data\raw\ 2>nul
move *.jsonl data\raw\ 2>nul
move *.xlsx data\raw\ 2>nul

REM Requisitos
move requirements*.txt config\ 2>nul
move user_agents.txt config\ 2>nul
move proxies.txt config\ 2>nul

REM Scrapy project (ya existe carpeta)
if exist "scrapy_employment_scraper" (
    echo [✔] Proyecto Scrapy detectado.
)

echo.
echo ✅ Organización completada con éxito.
echo ------------------------------------------
echo Puedes revisar: scripts\, data\, models\, img\, config\, docs\
echo ------------------------------------------

pause
