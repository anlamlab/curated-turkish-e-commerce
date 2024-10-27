"""
preprocessing.py

This script processes various datasets for sentiment analysis, cleaning and standardizing
the data format. It supports processing individual datasets and unifying multiple processed datasets.
For more info about the source of datasets, see the README.md file.
Supported datasets: winvoker, gorengoz, vscr, trsav
"""

import argparse
import pandas as pd
from pathlib import Path
import json
import re
from sklearn.model_selection import train_test_split


def split_dataset(unified_path):
    """
    Split the dataset into train and test sets, ensuring equal representation from each dataset.

    Args:
        unified_path (String): The input path to split.
    """

    if not unified_path.exists():
        print("Error: Unified dataset with ChatGPT labels not found. Please run with -append-chatgpt first.")
        return
    df = pd.read_csv(unified_path)

    train_dfs = []
    test_dfs = []

    for dataset in df['Source_Dataset'].unique():
        dataset_df = df[df['Source_Dataset'] == dataset]
        # Change test  size if you want a different split than 10-90 split. Currently test_size=0.1
        dataset_train, dataset_test = train_test_split(dataset_df, test_size=0.1, random_state=42,
                                                       stratify=dataset_df['chatgpt4mini'])
        train_dfs.append(dataset_train)
        test_dfs.append(dataset_test)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Save train and test datasets
    output_dir = Path('datasets/processed_split')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'unified_with_gpt_label_train.csv'
    test_path = output_dir / 'unified_with_gpt_label_test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train dataset saved as: {train_path}")
    print(f"Test dataset saved as: {test_path}")
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")


def clean_text(text):
    """
    Clean and standardize input text.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned and standardized text.
    """
    text = str(text) if pd.notna(text) else ''
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = text.replace('\\n', ' ')
    text = re.sub(r' +', ' ', text)
    return text.strip()


def safe_clean_text(text):
    """
    Safely clean text, handling potential errors.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text or original text if cleaning fails.
    """
    try:
        return clean_text(text)
    except Exception as e:
        print(f"Error cleaning text: {text}")
        print(f"Error message: {str(e)}")
        return text


def get_sentiment_abbr(label):
    """
    Get abbreviated sentiment label.

    Args:
        label (str): Full sentiment label.

    Returns:
        str: Abbreviated sentiment label.
    """
    if label == 'Positive':
        return 'pos'
    elif label == 'Negative':
        return 'neg'
    elif label == 'Neutral':
        return 'neu'
    else:
        return 'unk'


def determine_label(star):
    """
    Determine sentiment label based on star rating.

    Args:
        star (int): Star rating.

    Returns:
        str: Sentiment label.
    """
    if star <= 2:
        return 'Negative'
    elif star >= 4:
        return 'Positive'
    else:
        return 'Neutral'


def get_dataset_name(source):
    """Extract the dataset name from the source."""
    return source.split('-')[0]


def process_winvoker(file_paths):
    """
    Process Winvoker dataset files.

    Args:
        file_paths (list): List of file paths to process.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, encoding='utf-8')
        is_train = 'train' in file_path.name.lower()
        source_prefix = 'winvoker-train-' if is_train else 'winvoker-test-'
        df['Source'] = source_prefix + df['dataset'].str.split('_').str[0]
        df['SourceID'] = range(1, len(df) + 1)
        print(f"Processing {source_prefix}, Record Number: {len(df)}")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.rename(columns={'text': 'Text', 'label': 'OriginalLabel'})

    required_columns = ['Text', 'OriginalLabel', 'dataset', 'Source', 'SourceID']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing the following required column(s): {', '.join(missing_columns)}")

    combined_df['Text'] = combined_df['Text'].apply(safe_clean_text)
    combined_df['UniqueID'] = combined_df.apply(
        lambda row: f"{row['Source']}-{get_sentiment_abbr(row['OriginalLabel'])}-{row['SourceID']}",
        axis=1
    )

    filtered_df = combined_df[combined_df['dataset'].isin(['urun_yorumlari', 'magaza_yorumlari'])]
    filtered_df = filtered_df.drop_duplicates(subset='Text', keep='first')

    final_df = filtered_df[['Source', 'SourceID', 'UniqueID', 'Text', 'OriginalLabel']].copy()
    final_df.reset_index(drop=True, inplace=True)
    final_df['ID'] = final_df.index + 1
    final_df = final_df[['ID', 'Source', 'SourceID', 'UniqueID', 'Text', 'OriginalLabel']]
    print(f"Final DataFrame shape: {final_df.shape}")
    return final_df


def process_gorengoz(file_paths):
    """
    Process Gorengoz dataset files.

    Args:
        file_paths (list): List of file paths to process.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    train_dfs = []
    test_dfs = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path)

        required_columns = ['description', 'durum']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Dataset must contain the '{column}' column")

        label_map = {'olumlu': 'Positive', 'olumsuz': 'Negative', 'nötr': 'Neutral'}
        df['OriginalLabel'] = df['durum'].map(label_map)
        is_train = 'train' in file_path.name.lower()
        df['Source'] = 'gorengoz-train' if is_train else 'gorengoz-test'
        df['SourceID'] = range(1, len(df) + 1)

        print(f"Processing {'gorengoz-train' if is_train else 'gorengoz-test'}, Record Number: {len(df)}")

        if is_train:
            train_dfs.append(df)
        else:
            test_dfs.append(df)

    combined_df = pd.concat(train_dfs + test_dfs, ignore_index=True)

    combined_df['UniqueID'] = combined_df.apply(
        lambda row: f"{row['Source']}-{get_sentiment_abbr(row['OriginalLabel'])}-{row['SourceID']}",
        axis=1
    )

    combined_df.rename(columns={'description': 'Text'}, inplace=True)
    combined_df['Text'] = combined_df['Text'].apply(safe_clean_text)
    combined_df.drop_duplicates(subset='Text', keep='first', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df['ID'] = combined_df.index + 1

    final_df = combined_df[['ID', 'Source', 'SourceID', 'UniqueID', 'Text', 'OriginalLabel']]

    print(f"Final DataFrame shape: {final_df.shape}")
    return final_df


def process_vscr(file_paths):
    """
    Process VSCR dataset file.

    Args:
        file_paths (list): List containing a single file path to process.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    if len(file_paths) > 1:
        raise ImportError(
            f"VSCR dataset preprocessing function is expecting single file. We detected {len(file_paths)} files.\n "
            f"Please check out 'datasets/original/vscr' directory.")

    with open(file_paths[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for product in data:
        product_id = product['id']
        reviews = product.get('reviews', [])

        for review in reviews:
            review_id = review['review_id']
            star = review['star']
            text = review['review']

            clean_review = clean_text(text)

            row = {
                'ID': len(rows) + 1,
                'Source': 'vscr',
                'SourceID': review_id,
                'UniqueID': f'vscr-prod-{product_id}-{get_sentiment_abbr(determine_label(star))}-{review_id}',
                'Text': clean_review,
                'OriginalLabel': determine_label(star),
            }

            rows.append(row)

    final_df = pd.DataFrame(rows)
    print(f"Processing TRSAV, Record Number: {len(final_df)}")
    final_df.drop_duplicates(subset='Text', keep='first', inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    final_df['ID'] = final_df.index + 1
    print(f"Final DataFrame shape: {final_df.shape}")

    return final_df


def process_trsav(file_paths):
    """
    Process TRSAV dataset file.

    Args:
        file_paths (list): List containing a single file path to process.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    if len(file_paths) != 1:
        raise ImportError(
            f"TRSAV dataset preprocessing function is expecting a single file. We detected {len(file_paths)} files.\n"
            f"Please check the 'datasets/original/trsav' directory.")

    df_trsav = pd.read_csv(file_paths[0], encoding='utf-8')
    print(f"Processing TRSAV, Record Number: {len(df_trsav)}")

    df_trsav['Source'] = 'trsav'
    df_trsav['SourceID'] = df_trsav['id'] if 'id' in df_trsav.columns else range(1, len(df_trsav) + 1)

    df_trsav = df_trsav.rename(columns={'review': 'Text', 'score': 'OriginalLabel'})

    required_columns = ['Text', 'OriginalLabel', 'Source', 'SourceID']
    missing_columns = [col for col in required_columns if col not in df_trsav.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing the following required column(s): {', '.join(missing_columns)}")

    df_trsav['Text'] = df_trsav['Text'].apply(clean_text)
    df_trsav['UniqueID'] = df_trsav.apply(
        lambda row: f"{row['Source']}-{get_sentiment_abbr(row['OriginalLabel'])}-{row['SourceID']}",
        axis=1
    )

    df_trsav.drop_duplicates(subset='Text', keep='first', inplace=True)
    df_trsav.reset_index(drop=True, inplace=True)

    final_df = df_trsav[['Source', 'SourceID', 'UniqueID', 'Text', 'OriginalLabel']].copy()
    final_df['ID'] = final_df.index + 1
    final_df = final_df[['ID', 'Source', 'SourceID', 'UniqueID', 'Text', 'OriginalLabel']]
    print(f"Final DataFrame shape: {final_df.shape}")
    return final_df


def unify_datasets(processed_dir):
    """
    Unify all processed datasets into a single file. The output will be written in 'datasets/processed_unified'

    Args:
        processed_dir (Path): Directory containing processed dataset files.
    """
    all_files = [f for f in processed_dir.glob('*.csv') if not f.stem.startswith('unified')]

    if not all_files:
        print("No individual dataset files found to unify.")
        return

    dfs = [pd.read_csv(f) for f in all_files]
    unified_df = pd.concat(dfs, ignore_index=True)

    dataset_names = [f.stem for f in all_files]
    unified_filename = f"unified_dataset_{'_'.join(dataset_names)}.csv"
    processed_dir_unified = Path('datasets/processed_unified')
    unified_path = processed_dir_unified / unified_filename

    unified_df.reset_index(drop=True, inplace=True)
    unified_df['ID'] = unified_df.index + 1
    unified_df['SourceDataset'] = unified_df['Source'].apply(get_dataset_name)
    unified_df = unified_df[['ID', 'SourceDataset', 'Source', 'SourceID', 'UniqueID', 'Text', 'OriginalLabel']]
    unified_df.to_csv(unified_path, index=False)
    print(f"Unified dataset saved as {unified_path}")
    print(f"Number of rows in unified dataset: {len(unified_df)}")


def append_chatgpt_label():
    """
    Append ChatGPT labels to the unified dataset using optimized pandas operations.

    Returns:
        None
    """
    # Step 1: Load the unified dataset
    unified_dir = Path('datasets/processed_unified')
    unified_files = list(unified_dir.glob('unified_dataset_*.csv'))
    if not unified_files:
        print("No unified dataset found.")
        return
    unified_df = pd.read_csv(unified_files[0])

    # Step 2: Load label mappings
    label_map_dir = Path('datasets/gpt_label_map')
    label_map_dfs = [pd.read_csv(file) for file in label_map_dir.glob('*_revised_labels.csv')]
    unified_label_map_df = pd.concat(label_map_dfs, ignore_index=True)

    # Step 3 & 4: Merge ChatGPT labels with unified dataset
    unified_df = unified_df.merge(unified_label_map_df[['UniqueID', 'chatgpt4mini']],
                                  on='UniqueID',
                                  how='left')

    # Fill NaN values in chatgpt4mini column with an empty string
    unified_df['chatgpt4mini'] = unified_df['chatgpt4mini'].fillna('')

    # Save the result
    output_path = 'datasets/processed_gpt_label/unified_with_gpt_label.csv'
    unified_df.to_csv(output_path, index=False)

    # Print overall statistics
    total_labels = (unified_df['chatgpt4mini'] != '').sum()
    print("Total number of unified dataset rows: ", len(unified_df))
    print(f"Total number of chatgptlabel appended is: {total_labels}")

    # Print statistics for each dataset
    dataset_stats = unified_df.groupby('Source_Dataset').apply(lambda x: (x['chatgpt4mini'] != '').sum()).sort_values(
        ascending=False)
    print("\nNumber of chatgptlabel appended by Dataset:")
    for dataset, count in dataset_stats.items():
        print(f"{dataset}: {count}")

    print(f"\nNew file saved as: {output_path}")


def main():
    """
    Main function to process datasets, optionally unify them, append ChatGPT labels, and split the dataset.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess datasets, optionally unify them, append ChatGPT labels, and split the dataset.")
    parser.add_argument('-dataset', type=str, help="Name of the dataset to process")
    parser.add_argument('-unify', action='store_true', help="Unify all processed datasets")
    parser.add_argument('-append-chatgpt', action='store_true', help="Append ChatGPT labels to the unified dataset")
    parser.add_argument('-split', action='store_true',
                        help="Split the unified chatgpt labelled dataset into train and test sets")
    args = parser.parse_args()

    original_dir = Path('datasets/original')
    processed_dir = Path('datasets/processed')
    unified_path = Path('datasets/processed_gpt_label/unified_with_gpt_label.csv')

    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_dir = original_dir / args.dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            print(f"Error: Dataset directory {args.dataset} not found in {original_dir}")
            return

        file_paths = [f for f in dataset_dir.iterdir() if
                      f.is_file() and f.suffix.lower() in ('.csv', '.parquet', '.json')]

        if not file_paths:
            print(f"Error: No CSV or Parquet files found in {dataset_dir}")
            return

        print(f"Processing files: {', '.join(str(f) for f in file_paths)}")

        if args.dataset == 'winvoker':
            processed_df = process_winvoker(file_paths)
        elif args.dataset == 'gorengoz':
            processed_df = process_gorengoz(file_paths)
        elif args.dataset == 'vscr':
            processed_df = process_vscr(file_paths)
        elif args.dataset == 'trsav':
            processed_df = process_trsav(file_paths)
        else:
            print(f"Error: No processing function defined for dataset {args.dataset}")
            return

        output_path = processed_dir / f"{args.dataset}.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"Processed dataset saved as {output_path}")

    if args.unify:
        unify_datasets(processed_dir)

    if args.append_chatgpt:
        append_chatgpt_label()

    if args.split:
        split_dataset(unified_path)


if __name__ == "__main__":
    main()
