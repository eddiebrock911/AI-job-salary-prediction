import pandas as pd

try:
    df = pd.read_csv('ai_job_dataset.csv')
    print("COLUMNS:")
    print(df.columns.tolist())
    print("\nDTYPES:")
    print(df.dtypes)
    print("\nHEAD(3):")
    print(df.head(3))
    print("\nNULLS:")
    print(df.isnull().sum())
    print("\nDESCRIBE:")
    print(df.describe())
except Exception as e:
    print(f"Error: {e}")
