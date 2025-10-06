import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_housing_data():
    try:
        data = pd.read_csv("cintia2025\\datasets\\housing\\housing.csv")
    except FileNotFoundError:
        data = pd.read_csv("datasets\\housing\\housing.csv")

    return data

def plot_histograms(data, figsize=(14, 8), bins=50, color="#5f4db1"):
    """
    data: Es el dataframe que contiene los datos de housing. Se supone que es el dataframe
    original, con todas las columnas.
    """
    data.hist(figsize=figsize, bins=bins, color=color)
    plt.show()
    return None