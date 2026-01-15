import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_data(raw_data_path="data/mushrooms_raw.csv"):
    """
    Loads the Mushroom dataset.
    Returns: pd.DataFrame of complete dataset (features + target).
    """
    
    if os.path.exists(raw_data_path):
        print(f"Loading data from local file: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        return df
    
    print("Fetching data from UCI Repository...")
    
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
  
    X = mushroom.data.features # type: ignore 
    y = mushroom.data.targets  # type: ignore

    df = pd.concat([X, y], axis=1)
        
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        
    # save to CSV for future use
    df.to_csv(raw_data_path, index=False)
    print(f"Data successfully fetched and saved to {raw_data_path}")
        
    return df

def drop_constant_columns(df):
    """
    Identifies and drops columns that have only one unique value across all rows.
    These columns have variance = 0 provide no information for association rules.
    Args:
        df (pd.DataFrame): The input dataframe.
    Returns:
        pd.DataFrame: Dataframe with constant columns removed.
    """
    # find columns with only 1 unique value
    unique_counts = df.nunique()
    cols_to_drop = unique_counts[unique_counts == 1].index.tolist()
    
    if len(cols_to_drop) > 0:
        print(f"[Data Cleaning] Dropping constant columns (no information): {cols_to_drop}")
        df_cleaned = df.drop(columns=cols_to_drop)
        return df_cleaned
    else:
        print("[Data Cleaning] No constant columns found.")
        return df

if __name__ == "__main__":
    # test to verify if it works when running this script directly
    df = load_data()
    if df is not None:
        print(f"Data loaded! Shape: {df.shape}")
        print(df.head())