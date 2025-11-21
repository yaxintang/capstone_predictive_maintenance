from data_pipeline.extract import list_csv_files, extract_csv
from data_pipeline.transform import transform
from data_pipeline.load import load_to_duckdb
import os
import duckdb

RAW_FOLDER = 'data/raw'
DUCKDB_FILE = 'data/team_data.duckdb'

def run_pipeline():
    # Prüfen, ob der raw-Ordner existiert
    if not os.path.exists(RAW_FOLDER):
        raise FileNotFoundError(f"Raw-Ordner existiert nicht: {RAW_FOLDER}")

    csv_files = list_csv_files(RAW_FOLDER)
    
    if not csv_files:
        print(f"Keine CSV-Dateien im Ordner {RAW_FOLDER} gefunden. Pipeline wird beendet.")
        return

    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # Dateiname ohne .csv
        #print(f"\nVerarbeite Datei: {file_path}")
        print(f"\nProcessing files: {file_path}")
        
        # Extract
        df = extract_csv(file_path)
        
        # Transform
        df_clean = transform(df, file_name)
        
        # Load
        load_to_duckdb(df_clean, DUCKDB_FILE, table_name=file_name)
    
    # print("\nPipeline abgeschlossen!")
    # print("Erstellte Tabellen in DuckDB:")

    print("\nPipeline finished!")
    print("Created Tables in DuckDB:")

    with duckdb.connect(DUCKDB_FILE) as conn:
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            print(f" - {table[0]}")

if __name__ == "__main__":
    run_pipeline()