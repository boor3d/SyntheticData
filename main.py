# main.py
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from data_ingestion.dataset_loader import load_huggingface_dataset
from data_preprocessing.image_preprocessing import image_preprocess, batch_dataset_to_dataframe
from generative_models.GAN.train_gan import train
import warnings
warnings.filterwarnings('ignore')

def main():

    parser = argparse.ArgumentParser(description='Load datasets from local file, library, or URL')
    parser.add_argument('--source', type=str, choices=['local', 'library', 'url'], required=True, help='Dataset source type')
    parser.add_argument('--file_path', type=str, help='Path to local dataset file (for "local" source)')
    parser.add_argument('--dataset_name', type=str, help='Name of library dataset (for "library" source)')
    parser.add_argument('--url', type=str, help='URL of dataset (for "url" source)')
    parser.add_argument('--datatype', type=str, choices=["audio", "image", "text", "time-series", "tabular"], required=True, help='Data Format Desired (audio, image, time-series, tabular)')

    args = parser.parse_args()

    data = None

    if args.source == 'library':


        data = load_huggingface_dataset(args.dataset_name, library=True)
        
        original_data = data.sample(n=1000, replace=False)

        # Write to csv for analysis 
        original_data.to_csv("./notebooks/data/data.csv", index=False)

        

        if args.datatype == "image":
            train_data = image_preprocess(original_data)


            train_data_df = batch_dataset_to_dataframe(train_data)
            # Write to csv for analysis 
            train_data_df.to_csv("./notebooks/data/train_data.csv", index=False)


            # Batch and shuffle the data
            buffer_size = 10
            batch_size = 32
            train_dataset = train_data

            print(f"Training Data: {train_data}")

            # Training parameters
            epochs = 300
            noise_dim = 100

            # Start training
            train(train_dataset, epochs, batch_size, noise_dim)


    return


if __name__ == "__main__":
    main()