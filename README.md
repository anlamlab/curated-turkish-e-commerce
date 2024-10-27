# Data in Brief- A Consolidated Turkish E-Commerce Review Dataset for Sentiment Analysis Using LLM Labelling

## How to Preprocess Public Datasets

This section provides instructions on how to use the `preprocessing.py` script to prepare datasets for further processing.

### Installation

1. If you don't have Python installed, Install Python 3.10 or later by following the official website below:
   - [Official Python Downloads](https://www.python.org/downloads/)
   - For detailed installation instructions, refer to the [Python Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download)

2. Install required libraries:
```pip install -r requirements.txt```

### Dataset Preparation

1. Download the original datasets from the following sources:
- TRSAv1: [TRSAv1 Dataset](https://huggingface.co/datasets/maydogan/TRSAv1/resolve/main/TRSAv1.csv)
- VSCR: [Vitamins-Supplements-Reviews Dataset](https://github.com/turkish-nlp-suite/Vitamins-Supplements-Reviews/blob/main/urunler-yorumlar.json)
- Winvoker:
     - Train: [Turkish Sentiment Analysis Train Dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset/resolve/main/train.csv)
     - Test: [Turkish Sentiment Analysis Test Dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset/resolve/main/test.csv)
- Gorengoz:
  - Train: [Customer Review Train Dataset](https://huggingface.co/datasets/Gorengoz/tr-customerreview/resolve/main/data/train-00000-of-00001.parquet)
  - Validation: [Customer Review Validation Dataset](https://huggingface.co/datasets/Gorengoz/tr-customerreview/resolve/main/data/validation-00000-of-00001.parquet)

 > ##### Credit and Acknowledgement
 > We would like to express our sincere gratitude to all the individuals and organizations who have made these datasets publicly available.

2. Place the downloaded datasets in their respective directories. Please note that if you don't want a particular file in the final preprocessed dataset, you can exclude it (e.g., you can only include train.csv under winvoker if you prefer):
```
    datasets/original/winvoker/
    datasets/original/gorengoz/
    datasets/original/vscr/
    datasets/original/trsav/
```

For example VSCR dataset expext this file under the bespoke directory ``` datasets/original/vscr/urunler-yorumlar.json``` .

 
### Preprocessing Steps

#### Winvoker Dataset
- Processes both train and test CSV files if available
- Combines train and test data, with train data appearing first
- Cleans text by removing newlines, quotes, and extra spaces
- Filters for 'urun_yorumlari' and 'magaza_yorumlari' categories
- Removes duplicate entries based on the 'Text' column

#### Gorengoz Dataset
- Processes Parquet files (train and validation)
- Maps 'durum' labels to 'Positive', 'Negative', 'Neutral'
- Combines train and validation data
- Removes duplicate entries based on the 'Text' column

#### VSCR Dataset
- Processes a single JSON file containing product reviews
- Extracts review information including product ID, review ID, star rating, and text
- Cleans review text by removing newlines and extra spaces
- Determines sentiment label based on star rating
- Creates a unique ID for each review
- Removes duplicate entries based on the 'Text' column

#### TRSAV Dataset
- Processes a single CSV file
- Renames 'review' column to 'Text' and 'score' column to 'OriginalLabel'
- Cleans text by removing newlines and extra spaces
- Creates a unique ID for each review
- Removes duplicate entries based on the 'Text' column

### Usage

To preprocess a specific dataset:

```
python preprocessing.py -dataset [dataset_name]
```

Replace `[dataset_name]` with one of: `winvoker`, `gorengoz`, `vscr`, or `trsav`.

Example:
```
python preprocessing.py -dataset winvoker
```

To unify all processed datasets:

```
python preprocessing.py -unify
```

This will combine all preprocessed datasets into a single CSV file.

<hr>

Additionally, if you want to append `ChatGPT4omini` sentiment analysis labels to the unified file created by using the command above, run following command which will append a column to the file. 

```
python preprocessing.py -append-chatgpt
```

The `-append-chatgpt` option performs the following actions:
1. Loads the unified dataset (if it exists).
2. Loads ChatGPT label mappings from CSV files in the `datasets/gpt_label_map` directory.
3. Merges the ChatGPT labels with the unified dataset based on the `UniqueID` column.
4. Saves the result as a new CSV file named `unified_with_gpt_label.csv` in the `datasets/processed_gpt_label` directory.
5. Prints statistics about the number of ChatGPT labels appended, both overall and for each dataset.

This option is useful for adding machine-generated sentiment labels to your dataset, which can be used for comparison or as additional features in your analysis.
 

To Split your dataset into train and test in proportion of 90%-10% respectively, run following command which will randomly pick stratified chatgpt40mini distribution of dataset 

``` 
python preprocessing.py -split

```

You can always change the proportion of train-test in the following line in the code by changing the value of test-size parameter which is currently set to 0.1.

``` 
dataset_train, dataset_test = train_test_split(dataset_df, test_size=0.1, random_state=42,
                                                       stratify=dataset_df['chatgpt4mini'])
```
## Credit and Acknowledgement

We would like to express our sincere gratitude to all the individuals and organizations who have made these datasets publicly available. Your contributions to the research community are invaluable and greatly appreciated. This project would not be possible without your commitment to open data and advancing the field of natural language processing.