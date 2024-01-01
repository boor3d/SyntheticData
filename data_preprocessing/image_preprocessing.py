import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import pandas as pd

def image_preprocess(df, batch_size=64, normalization_range=(-1, 1)):
    # Check if images are numpy arrays and normalize them
    normalized_images = []
    for image in df['image'].values:
        if isinstance(image, np.ndarray):
            normalized_image = (image - normalization_range[0]) / (normalization_range[1] - normalization_range[0])
            normalized_images.append(normalized_image)
        else:
            raise TypeError("Images must be numpy arrays.")

    # Convert the list of normalized images to a numpy array
    normalized_images = np.array(normalized_images)

    # Convert the numpy array of images to a TensorFlow dataset and batch it
    train_dataset = tf.data.Dataset.from_tensor_slices(normalized_images).batch(batch_size)

    return train_dataset



def batch_dataset_to_dataframe(batch_dataset):
    images = []
    labels = []

    # Iterate through the BatchDataset
    for batch in batch_dataset:
        print (batch)
        break
        batch_images, batch_labels = batch
        # Flatten the images if necessary and add to the list
        images.extend(batch_images.numpy().reshape(batch_images.shape[0], -1))  # Reshaping to flatten the images
        labels.extend(batch_labels.numpy())

    # Create a DataFrame
    df = pd.DataFrame(images)
    df['label'] = labels  # Assuming 'labels' are your target values

    return df
