import pyodbc

try:
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=W5CG2241T4M\\MSSQLSERVER01;"
        "DATABASE=CloudSQL;"
        "Trusted_Connection=yes;"
    )
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION")
    version = cursor.fetchone()[0]
    print(f"Connected successfully using Windows authentication: {version}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")