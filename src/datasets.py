"""Loaders for the two datasets used in the project."""

import csv
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CALIFORNIA_CSV = DATA_DIR / "california_housing_train.csv"
CALIFORNIA_URL = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
CALIFORNIA_TARGET = "median_house_value"


def load_california_housing(path=CALIFORNIA_CSV):
    """California Housing dataset (1990 U.S. census).

    17,000 census districts described by 8 numerical features (location,
    age of the houses, rooms, bedrooms, population, households, income); the
    target is the median house value of the district.

    The CSV file is the one distributed with Google Colab
    (``sample_data/california_housing_train.csv``). If it is not found at
    ``path`` it is downloaded once from the Google ML Crash Course server.

    Returns
    -------
    X : ndarray of shape (n_samples, 8)
    y : ndarray of shape (n_samples,)
    feature_names : list of str
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(CALIFORNIA_URL, path)
        except OSError as error:
            raise FileNotFoundError(
                f"{path} not found and the download failed ({error}). "
                f"Download it manually from {CALIFORNIA_URL} and place it in {path.parent}."
            ) from error

    with open(path, newline="") as file:
        header = next(csv.reader(file))
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    target_column = header.index(CALIFORNIA_TARGET)
    feature_columns = [i for i in range(len(header)) if i != target_column]
    return (data[:, feature_columns], data[:, target_column],
            [header[i] for i in feature_columns])


def load_breast_cancer():
    """Breast Cancer Wisconsin (Diagnostic) dataset, as shipped with scikit-learn.

    569 tumours described by 30 numerical features computed from a digitised
    image of a cell sample; the label is 0 for malignant and 1 for benign.

    Returns
    -------
    X : ndarray of shape (569, 30)
    y : ndarray of shape (569,) with values in {0, 1}
    feature_names : list of str
    """
    from sklearn.datasets import load_breast_cancer as sklearn_loader

    dataset = sklearn_loader()
    return dataset.data, dataset.target, list(dataset.feature_names)
