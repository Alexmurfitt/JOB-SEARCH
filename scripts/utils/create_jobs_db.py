import psycopg2

try:
    # Conexión al servidor PostgreSQL para crear la base
    conn = psycopg2.connect(
        dbname='postgres',  # base de administración por defecto
        user='postgres',
        password='Alex.71087',
        host='localhost'
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE DATABASE jobs_db;")
    print("✅ Base de datos 'jobs_db' creada correctamente.")
    cur.close()
    conn.close()
except psycopg2.errors.DuplicateDatabase:
    print("⚠️ La base de datos 'jobs_db' ya existe.")
except Exception as e:
    print("❌ Error al crear la base de datos:", type(e).__name__, str(e))
