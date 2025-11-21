import duckdb
import os
import re

def clean_table_name(file_name):
    """
    Säubert den Tabellennamen:
    - Ersetzt alle Sonderzeichen durch Unterstrich _
    - Entfernt Leerzeichen
    """
    # Nur Buchstaben, Zahlen und Unterstrich erlauben
    return re.sub(r'\W+', '_', file_name)

def load_to_duckdb(df, duckdb_path='data/team_data.duckdb', table_name='default_table'):
    """
    Speichert ein DataFrame in DuckDB.
    Tabelle wird überschrieben, Tabellennamen automatisch gesäubert.
    """
    # Tabellennamen säubern
    table_name = clean_table_name(table_name)
    
    # Ordner für DuckDB-Datei erstellen, falls nicht vorhanden
    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)
    
    # Verbindung zu DuckDB aufbauen
    conn = duckdb.connect(duckdb_path)
    
    # Tabelle überschreiben, falls sie existiert
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.register('df_temp', df)
    conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_temp')
    
    print(f"Geladen in {duckdb_path} -> Tabelle: {table_name}")
    conn.close()