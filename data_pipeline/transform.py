import pandas as pd

def transform(df, file_name=''):
    """
    Bereinigt Daten und erstellt neue Spalten.
    Aber für unsere wird hier nichts gemacht
    Denn alles dann in EDA passieren wird
    Hier werden nur basis Prüfungen machen
    """
    # prüfe , dass df not leer ist
    if len(df) != 0:
        df_clean = df
    
    #print(f"Transformiert: {len(df_clean)} Zeilen für {file_name}")
    print(f"Transformed: {len(df_clean)} rows for {file_name}")
    return df_clean