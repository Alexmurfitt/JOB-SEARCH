@echo off
echo 📦 Iniciando limpieza de estructura del proyecto...

rem Archivos individuales
move "scripts\estructura_documentada_extendida.txt" "docs"
move "scripts\estructura_final.txt" "docs"
move "scripts\estructura_resumida.md" "docs"
move "scripts\metrics.json" "reports"
move "scripts\metrics_train.json" "reports"

rem Carpetas completas
move "scripts\artifacts" "artifacts"
move "scripts\catboost_info" "models\catboost_info"
move "scripts\models" "models"
move "scripts\utils" "scripts\utils"

rem Eliminar carpeta anidada errónea
rmdir /s /q "scrapy_employment_scraper\spiders\scrapy_employment_scraper"

echo ✅ Limpieza completada.
pause
