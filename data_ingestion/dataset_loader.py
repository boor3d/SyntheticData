import pandas as pd
import os
import requests
import mimetypes
from io import BytesIO
from PIL import Image
from sklearn import datasets as sk_datasets

def load_data(source, data_type=None, sklearn_dataset=False):
    """
    Load data from a local file, a URL, or sklearn datasets.
    
    :param source: URL, local file path, or sklearn dataset name.
    :param data_type: Type of data (e.g., 'csv', 'json', 'image'). If None, attempt to infer.
    :param sklearn_dataset: Boolean, set to True if loading a dataset from sklearn.
    :return: Loaded data and inferred type.
    """
    if sklearn_dataset:
        if hasattr(sk_datasets, source):
            dataset = getattr(sk_datasets, source)()
            return pd.DataFrame(data=dataset.data, columns=dataset.feature_names), 'tabular'
        else:
            raise ValueError(f"Sklearn dataset '{source}' not found.")

    if source.startswith('http://') or source.startswith('https://'):
        response = requests.get(source)
        content_type = response.headers['content-type']
        if 'text' in content_type or 'json' in content_type:
            return pd.read_csv(BytesIO(response.content)), 'tabular'
        elif 'image' in content_type:
            return Image.open(BytesIO(response.content)), 'image'
        else:
            raise ValueError("Unsupported data type from URL")

    elif os.path.isfile(source):
            mime_type, _ = mimetypes.guess_type(source)
            if mime_type and 'text' in mime_type:
                return pd.read_csv(source), 'tabular'
            elif mime_type and 'image' in mime_type:
                return Image.open(source), 'image'
            else:
                raise ValueError("Unsupported file type")

    else:
        raise ValueError("Unsupported data source or file not found")

    raise ValueError("Unable to load data from the source")