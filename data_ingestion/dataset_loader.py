import tensorflow_datasets as tfds
import pandas as pd

def load_dataset(dataset_name, library=False):
    if library:
        # Load the specified TensorFlow dataset
        dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)

        # Select a specific split, e.g., 'train'
        train_data = dataset['train'].take(10000)  # Adjust the number of samples as needed

        # Initialize lists for images and labels
        images, labels = [], []

        # Iterate over the dataset and convert to numpy arrays
        for image, label in tfds.as_numpy(train_data):
            images.append(image)
            labels.append(label)

        # Create a DataFrame
        df = pd.DataFrame({'image': images, 'label': labels})

        return df 