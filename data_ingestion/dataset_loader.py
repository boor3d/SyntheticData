from datasets import load_dataset
import pandas as pd
import numpy as np

def process_images(df):
    processed_images = []

    for image in df['image']:
        # Convert PIL image to numpy array and reshape to (28, 28, 1)
        image_array = np.array(image).reshape(28, 28, 1)
        processed_images.append(image_array)

    # Replace the 'image' column in the DataFrame
    df['image'] = processed_images
    return df

def load_huggingface_dataset(dataset_name, library=True):
    
    # Load the entire dataset
    dataset = load_dataset(dataset_name)

    # The dataset can be a DatasetDict with multiple splits, or a single Dataset
    if isinstance(dataset, dict):
        # If it's a DatasetDict, you can concatenate the splits, or choose one
        # For example, concatenate train and test splits
        dataset = pd.concat([pd.DataFrame(dataset[split]) for split in dataset.keys()])
    else:
        # Convert to pandas DataFrame
        dataset = pd.DataFrame(dataset)

    dataset_processed = process_images(dataset)

    return dataset_processed
