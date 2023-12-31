import argparse
from data_ingestion.dataset_loader import load_data

def main(args):
    # Load data based on arguments
    data, data_type = load_data(args.source, sklearn_dataset=args.sklearn_dataset)
    print(f"Data loaded: {data_type}")


if __name__ == "__main__":
    """
    python main.py --source load_iris --sklearn_dataset | Pass in sklearn dataset
    python main.py --source /path/to/your/dataset.csv | Pass in local dataset 
    python main.py --source http://example.com/data.csv | Pass in URL
    """
    parser = argparse.ArgumentParser(description="Data Loading Script")
    parser.add_argument('--source', type=str, help='Data source (URL, file path, or sklearn dataset name)')
    parser.add_argument('--sklearn_dataset', action='store_true', help='Set true if loading a sklearn dataset')

    args = parser.parse_args()
    main(args)
