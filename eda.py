import pandas as pd
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv(file_path):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def inspect_dataframe(df):
    """Inspect the first few rows and column names of the DataFrame."""
    print("First few rows of the DataFrame:")
    print(df.head())
    print("Column Names:", df.columns.tolist())

def extract_labels_from_column(df, column_name):
    """Extract and flatten labels from a specific DataFrame column."""
    all_labels = []
    for labels_str in df[column_name]:
        try:
            labels_list = ast.literal_eval(labels_str)
            all_labels.extend(labels_list)
        except (SyntaxError, ValueError) as e:
            logging.warning(f"Error parsing label string: {labels_str} - {e}")
            all_labels.append(labels_str)
    return all_labels

def get_unique_labels(all_labels):
    """Extract unique labels from a list of labels."""
    return list(set(all_labels))

def main():
    file_path = 'pii_dataset.csv'
    df = read_csv(file_path)
    inspect_dataframe(df)
    
    # Extract labels from the 'labels' column
    all_labels = extract_labels_from_column(df, 'labels')
    unique_labels = get_unique_labels(all_labels)
    
    print("Unique Labels from 'labels' column:", unique_labels)

if __name__ == "__main__":
    main()