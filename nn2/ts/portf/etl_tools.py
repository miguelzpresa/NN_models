import pandas as pd
import requests
import yfinance as yf
from pathlib import Path
from typing import Union, List, Optional
import os

def get_sp500_tickets():

  url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
  data = requests.get(url)
  sp500_tickets = pd.read_html(data.text)
  d = sp500_tickets[0]
  subset = d.select_dtypes(include='object').columns.tolist()
  d[subset] = d[subset].astype('string')
  return d

def object_to_string(df):
  """
  Convert all object columns in a DataFrame to string type.
  """
  subset = df.select_dtypes(include='object').columns.tolist()
  df[subset] = df[subset].astype('string')
  return df

def download_sp500_history(d,  start="2020-01-01", 
                            end="2021-12-31",
                            output_dir="data/sp500"):
    
    os.makedirs(output_dir, exist_ok=True)
    ticker_sp500 = "^GSPC"
    data_sp500_download = yf.download(ticker_sp500, start=start, end=end)
    data_sp500_download.to_csv(f"{output_dir}/sp500.csv")
    for ticket in d["Symbol"]:
        try:
            data = yf.download(ticket, start="2020-01-01", end="2021-12-31")
        except Exception as e:
            print(f"Error downloading data for {ticket}: {e}")
            continue
        data.to_csv(f"{output_dir}/{ticket}.csv")
        print(f"Downloaded data for {ticket}")

    




def load_financial_data(
    base_dir: Union[str, Path],
    main_file: str,
    additional_files: Optional[List[str]] = None,
    skip_rows: int = 3,
    date_col: str = "date",
    round_decimals: int = 4,
    save_output: bool = False,
    output_name: str = "combined_data.csv"
) -> pd.DataFrame:
    """
    Carga y combina archivos financieros en formato CSV con estructura similar.
    
    Args:
        base_dir: Directorio base donde se encuentran los archivos
        main_file: Archivo principal que contiene la columna de fechas
        additional_files: Lista de archivos adicionales a combinar (opcional)
        skip_rows: Filas a saltar al inicio del archivo (default: 3)
        date_col: Nombre para la columna de fechas (default: "date")
        round_decimals: Decimales para redondeo (default: 4)
        save_output: Si True, guarda el resultado (default: False)
        output_name: Nombre del archivo de salida (default: "combined_data.csv")
    
    Returns:
        DataFrame combinado con todos los datos
    """
    base_path = Path(base_dir)
    
    # Cargar archivo principal
    main_col = Path(main_file).stem  # Elimina la extensión
    df = pd.read_csv(
        base_path / main_file,
        skiprows=skip_rows,
        names=[date_col, main_col],
        parse_dates=[0],
        usecols=[0, 1],
        converters={main_col: lambda x: round(float(x), round_decimals)}
    )
    
    # Cargar archivos adicionales si existen
    if additional_files:
        for file in additional_files:
            col_name = Path(file).stem
            temp_df = pd.read_csv(
                base_path / file,
                skiprows=skip_rows,
                names=[col_name],
                usecols=[1],
                converters={col_name: lambda x: round(float(x), round_decimals)}
            )
            df[col_name] = temp_df[col_name]
    
    # Guardar si se solicita
    if save_output:
        df.to_csv(base_path / output_name, index=False)
        print(f"Datos guardados en: {base_path/output_name}")
    
    return df


def clean_and_save_data(df, save_path=None, name="cleaned_data.csv",null_threshold=1):
    """
    Elimina columnas con valores nulos y opcionalmente guarda el resultado.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar
        save_path (str, optional): Ruta donde guardar el archivo. Si es None, no guarda.
        name: nombre del archivo a guardar si save_path es especificado
        null_threshold (int): Umbral de valores nulos para eliminar columnas
        
    
    Returns:
        pd.DataFrame: DataFrame procesado sin columnas con muchos nulos
    """
    # Calcular nulos por columna y ordenar  
    null_counts = df.isnull().sum().sort_values(ascending=False)
    
    # Filtrar columnas que superan el umbral
    cols_to_drop = null_counts[null_counts > null_threshold].index.tolist()
    
    # Eliminar columnas problemáticas
    cleaned_df = df.drop(columns=cols_to_drop)
    
    # Guardar si se especificó path
    if save_path:
        cleaned_df.to_csv(f"{save_path}/{name}", index=False)
        print(f"Datos guardados en: {save_path}")
    
    return cleaned_df