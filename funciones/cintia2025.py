import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
import sklearn
sklearn.set_config(display="diagram")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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

def comparar_proporciones(raw_data, test_stratificado, test_purerandom):
    """
    raw_data, test_stratificado y test_purerandom deben contener la columna income_cat.
    Son dataframes.
    """
    ##chequeo que la columna exista
    if "income_cat" not in raw_data.columns:
        raise ValueError("La columna income_cat no está en raw_data")
    if "income_cat" not in test_stratificado.columns:
        raise ValueError("La columna income_cat no está en test_stratificado")
    if "income_cat" not in test_purerandom.columns:
        raise ValueError("La columna income_cat no está en test_purerandom")
    
    compare_props = pd.DataFrame({
    "Conjunto completo %": raw_data["income_cat"].value_counts(normalize=True).sort_index(),
    "Estraficado %": test_stratificado["income_cat"].value_counts(normalize=True).sort_index(),
    "Pure random %": test_purerandom["income_cat"].value_counts(normalize=True).sort_index(),
    })
    compare_props.index.name = "Income Category"
    compare_props["Strat. Error %"] = (compare_props["Estraficado %"] /
                                    compare_props["Conjunto completo %"] - 1)
    compare_props["Rand. Error %"] = (compare_props["Pure random %"] /
                                    compare_props["Conjunto completo %"] - 1)
    return (compare_props * 100).round(2)


def makeHousingScatter(housing, figsize=(12, 6)):
    housing_renamed = housing.rename(columns={
    "latitude": "Latitud", "longitude": "Longitud",
    "population": "Población",
    "median_house_value": "Valor medio de casas (USD)"})
    housing_renamed.plot(
                kind="scatter", x="Longitud", y="Latitud",
                s=housing_renamed["Población"] / 100, label="Población",
                c="Valor medio de casas (USD)", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=figsize)
    plt.title("Distribución de distritos en California", fontsize=12)
    try:
        california_img = plt.imread("cdan/imagenes/california.png")
    except FileNotFoundError:
        california_img = plt.imread("imagenes\\california.png")
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

    plt.show()

def cleanData(raw_data):
    """
    Función para limpiar los datos de housing.
    """
    # Manejo de valores nulos
    imputer = SimpleImputer(strategy="median")
    ##sólo podemos aplicar la mediana a columnas numéricas
    housing_num = raw_data.select_dtypes(include=[np.number])
    imputer.fit(housing_num)
    cleaned_data = imputer.transform(housing_num)
    return cleaned_data, imputer

class StandardScalerClone(BaseEstimator, TransformerMixin):
    """
    Esta clase es creada como un escalador estándar similar a
    sklearn.preprocessing.StandardScaler. Escala las características para que tengan media cero
    y desviación estándar uno. Si se establece with_mean en False, no se centrarán los datos.
    """
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # cada estimador debe guardar este atributo
        return self  # siempre devolver self!

    def transform(self, X):
        check_is_fitted(self)  # busca atributos aprendidos (con _ al final)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
    
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Esta clase crea una transformación basada en la similitud a los centroides de clusters
    obtenidos mediante KMeans. La similitud se calcula usando el kernel RBF (Radial
    Basis Function).
    """
    def __init__(self, n_clusters=10, gamma=1., random_state=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
def ratio_columnas(X):
    """
    Calcula la razón entre dos columnas del array X.
    """
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    """
    Genera un nombre para la nueva característica creada por la función
    function_transformer.
    """
    return ["ratio"]

def ratio_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("function_transformer", FunctionTransformer(ratio_columnas, feature_names_out=ratio_name)),
        ("scaler", StandardScaler()),
    ])

def get_FullProcesamiento():
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transformer", FunctionTransformer(np.log, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline_categorico = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Paso de imputación
    ("onehot", OneHotEncoder(handle_unknown="ignore")),  # Paso de codificación one-hot
    ])

    preprocesamiento = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),  # Nueva variable
        ("rooms_per_house",ratio_pipeline(), ["total_rooms", "households"]),  # Nueva variable
        ("people_per_house", ratio_pipeline(), ["population", "households"]),  # Nueva variable
        ("log", log_pipe, ["total_bedrooms", "total_rooms", "population", "households", "median_income"] ),  # Transformación logarítmica
        ("geo", cluster_simil, ["latitude", "longitude"]),  # Similitud geográfica
        ("cat", pipeline_categorico, make_column_selector(dtype_include=object)),  # One-hot encoding
    ], remainder=num_pipeline)  # El resto de columnas numéricas

    return preprocesamiento