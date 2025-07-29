
# Proyecto SEARCH_JOB

## Descripción
Pipeline completo para búsqueda y análisis de ofertas de empleo, entrenamiento y evaluación de modelos de predicción de salario.

## Estructura
- **config/**: archivos de configuración YAML  
- **config/schema/**: esquemas de columnas  
- **data/raw/**: datos originales  
- **data/processed/**: datos tras ETL  
- **data/splits/**: train/holdout  
- **scripts/**: código de ETL, entrenamiento y evaluación  
- **notebooks/**: análisis exploratorio  
- **tests/**: pruebas unitarias  
- **artifacts/**: modelos y figuras resultantes  
- **reports/**: informes finales  
- **.github/workflows/**: CI/CD  

## Cómo usar


# Crear estructura
generate_estructura_extendida.py [--force] [--dry-run]

# ETL
python scripts/etl.py --input data/raw --output data/processed

# Entrenar
python scripts/train_model.py --data data/processed --out artifacts/models

# Evaluar
python scripts/evaluate_model.py --model artifacts/models/model.pkl --test data/splits/holdout.csv --report artifacts/figures
