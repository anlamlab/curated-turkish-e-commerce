'''
 This file processes a chunk of the input file, for example, 1000 lines, and sends it to the chatgpt40-mini
 API for sentiment analysis. Due to the large context size, we sometimes need to reduce the chunk size and
 reprocess files, which is done in gpt_mini.py. This can result in issues with line order, which we corrected
  in rearrange_mini.py.

'''


import pandas as pd
import openai
from tqdm import tqdm
import os
from openai import OpenAI

# Set up OpenAI API key
openai.api_key = 'add-your-openai-key-here'

# Define file paths
input_file = 'datasets/processed_gpt_label/file_name.csv'
output_file = 'datasets/processed_gpt_label/output.csv'
client = OpenAI(api_key='add-your-openai-key-here')


# Function to get GPT labels for a batch of comments
def get_gpt_label(comment_dic):
    comments_str = "\n".join([f"id: {k}, comment: {v}" for k, v in comment_dic.items()])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user",
             "content": f"Please analyze the sentiment of the following review dictionary and return the result in the format 'id,label' where label should be one of these; Positive, Negative or Neutral:\n\n{comments_str}"}
        ]
    )
    # Parse the response to extract the sentiments
    results = response.choices[0].message.content.strip().split('\n')
    results = {line.split(',')[0]: str(line.split(',')[1]).strip() for line in results if 'Positive' in line or 'Negative' in line or 'Neutral' in line}
    return results


# Function to get the last processed ID
def get_last_processed_id(output_fil):
    if os.path.exists(output_fil) and os.stat(output_fil).st_size != 0:
        output_dat = pd.read_csv(output_fil)
        last_processed_i = output_dat['id'].iloc[-1]
        return last_processed_i
    return None




# Get the last processed ID
last_processed_id = get_last_processed_id(output_file)

# Read the input CSV file
data = pd.read_csv(input_file)

# If there's a last processed ID, filter the data to start from the next row
if last_processed_id is not None:
    data = data[data['id'] > last_processed_id]

# If the output file already exists, read it
if os.path.exists(output_file):
    output_data = pd.read_csv(output_file)
else:
    output_data = pd.DataFrame(columns=['id', 'source', 'comment', 'previous_label', 'chatgpt4mini'])

# Process the data in batches of  XXX
batch_size = 100
for i in tqdm(range(0, len(data), batch_size)):
    batch = data.iloc[i:i + batch_size]
    comment_dict = {str(row['id']): row['comment'] for idx, row in batch.iterrows()}
    try:
        sentiment_dict = get_gpt_label(comment_dict)
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        continue

    for idx, row in batch.iterrows():
        if str(row['id']) in sentiment_dict:
            row['chatgpt4mini'] = sentiment_dict[str(row['id'])]
            output_data = pd.concat([output_data, pd.DataFrame([row])], ignore_index=True)

    # Save progress
    output_data.to_csv(output_file, index=False)

    # Print status update
    print(f"Processed {i + batch_size} lines")

print("Processing complete.")
