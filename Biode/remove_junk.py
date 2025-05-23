import pandas as pd

def clean_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df_cleaned = df[~df.apply(lambda row: row.astype(str).str.contains('dicots', case=False)).any(axis=1)]

    return df_cleaned

cleaned_df = clean_dataset('dataset/dataset for rag/observations_with_tally.csv')
cleaned_df.to_csv('dataset/dataset for rag/cleaned_observations_with_tally.csv', index=False)
