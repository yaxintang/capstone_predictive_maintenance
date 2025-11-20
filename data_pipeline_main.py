from data_pipeline.extract import list_csv_files, extract_csv
from data_pipeline.transform import transform
from data_pipeline.load import load_to_duckdb
import os

RAW_FOLDER = 'data/raw'
DUCKDB_FILE = 'data/team_data.duckdb'

# csv_files = [os.path.join(RAW_FOLDER, f) for f in os.listdir(RAW_FOLDER) if f.endswith('.csv')]

# for file_path in csv_files:
#     df = extract_csv(file_path)
#     df_clean = transform(df)
#     table_name = os.path.splitext(os.path.basename(file_path))[0]
#     load_to_duckdb(df_clean, DUCKDB_FILE, table_name)

def run_pipeline():
    csv_files = list_csv_files(RAW_FOLDER)
    
    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # Dateiname ohne .csv
        df = extract_csv(file_path)
        df_clean = transform(df, file_name)
        load_to_duckdb(df_clean, DUCKDB_FILE, table_name=file_name)

if __name__ == "__data_pipeline_main__":
    run_pipeline()