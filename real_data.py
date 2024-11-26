import pandas as pd
from sklearn.datasets import load_wine, load_diabetes, load_breast_cancer, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def get_processed_wine_data():
    wine = load_wine()

    df_features = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df_target = pd.DataFrame(data=wine.target, columns=['target'])

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(df_features)

    data_by_samples = standardized_features
    target_by_samples = df_target.values

    return data_by_samples, target_by_samples


def get_processed_diabetes_data():
    diabetes = load_diabetes()

    df_features = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df_target = pd.DataFrame(data=diabetes.target, columns=['target'])

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(df_features)

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(standardized_features)

    return pca_features, df_target.values


def get_processed_iris_data():
    iris = load_iris()

    df_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_target = pd.DataFrame(data=iris.target, columns=['target'])

    df = pd.concat([df_features, df_target], axis=1)

    df = df.drop_duplicates()

    df_features = df.drop(columns=['target'])
    df_target = df['target']

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(df_features)

    data_by_samples = standardized_features
    target_by_samples = df_target.values

    return data_by_samples, target_by_samples


def get_processed_wdbc_data():
    wdbc = load_breast_cancer()

    df_features = pd.DataFrame(data=wdbc.data, columns=wdbc.feature_names)
    df_target = pd.DataFrame(data=wdbc.target, columns=['target'])

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(df_features)

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(standardized_features)

    return pca_features, df_target.values


def get_processed_wholesale_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
    df = pd.read_csv(url)

    df_features = df.drop(columns=['Channel', 'Region'])
    df_target = df[['Channel', 'Region']]

    duplicates = df.duplicated()
    print(f"Number of duplicate records: {duplicates.sum()}")

    df = df.drop_duplicates()

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(df_features)

    data_by_samples = standardized_features
    target_by_samples = df_target.values
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(data_by_samples)

    return pca_features, target_by_samples


def get_processed_glass_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    df = pd.read_csv(url, names=columns)

    df = df.drop(columns=['Id'])

    duplicates = df.duplicated()

    df = df.drop_duplicates()

    df_features = df.drop(columns=['Type'])
    df_target = df['Type']

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(df_features)

    data_by_samples = standardized_features
    target_by_samples = df_target.values

    return data_by_samples, target_by_samples


def get_processed_yeast_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    columns = ['Sequence_Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'Class']
    df = pd.read_csv(url, sep="\s+", names=columns)

    df = df.drop(columns=['Sequence_Name'])

    duplicates = df.duplicated()

    df = df.drop_duplicates()

    df_features = df.drop(columns=['Class'])
    df_target = df['Class']

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(df_features)

    data_by_samples = standardized_features
    target_by_samples = df_target.values

    return data_by_samples, target_by_samples

def get_processed_digits_data():
    digits = load_digits()
    data_by_samples = digits.data
    target_by_samples = digits.target

    scaler = StandardScaler()

    standardized_features = scaler.fit_transform(data_by_samples)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(standardized_features)

    return reduced_data, target_by_samples