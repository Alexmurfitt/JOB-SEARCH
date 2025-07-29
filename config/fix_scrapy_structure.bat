@echo off
REM Crear carpeta spiders si no existe
if not exist "scrapy_employment_scraper\spiders" (
    mkdir "scrapy_employment_scraper\spiders"
)

REM Mover el spider
move "scrapy_playwright_spiders\spiders\indeed_spider.py" "scrapy_employment_scraper\spiders\indeed_spider.py"

REM Crear scrapy.cfg
(
echo [settings]
echo default = scrapy_employment_scraper.settings
echo.
echo [deploy]
echo project = scrapy_employment_scraper
) > scrapy.cfg

echo ✅ Estructura actualizada correctamente. Ya puedes ejecutar: scrapy crawl indeed
pause
