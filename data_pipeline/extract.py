import pandas as pd
import os

def list_csv_files(raw_folder='data/raw'):
    """Listet alle CSV-Dateien im raw-Ordner auf."""
    return [
        os.path.join(raw_folder, f)
        for f in os.listdir(raw_folder)
        if f.lower().endswith('.csv')
    ]

def extract_csv(file_path):
    """Liest eine CSV-Datei ein und gibt ein Pandas DataFrame zurück."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} existiert nicht")
    df = pd.read_csv(file_path)
    #print(f"Extrahiert: {len(df)} Zeilen aus {file_path}")
    print(f"Extracted: {len(df)} rows from {file_path}")
    return df

