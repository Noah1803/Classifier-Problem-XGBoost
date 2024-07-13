import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from yellowbrick.model_selection import FeatureImportances
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scikitplot.metrics import plot_roc, plot_confusion_matrix, plot_precision_recall_curve
import joblib

def df_csv(nomedf, csv):
    nomedf = pd.read_csv(csv)
    return nomedf

def classe_boll_1(df):
    df['class'] = df['class'].replace('pos', 1)
    return df

def classe_boll_0(df):
    df['class'] = df['class'].replace('neg', 0)
    return df


def exchange_dtypes_NaN(df):

    columns_types = df.dtypes
    object_columns_previous = columns_types[columns_types == 'object'].index.tolist()
    object_columns_previous = object_columns_previous
    for col in object_columns_previous:
        # Substituir 'na' por NaN
        df[col] = df[col].replace('na', np.nan)
        try:
            # Tentar converter a coluna para float64
            df[col] = df[col].astype('float64')
        except ValueError:
            print(f"Coluna {col} não pôde ser convertida para float64.")

    print("\nDataFrame após a conversão:")
    print(df)
    return df

def fill_LR(df):
    target_col = [column for column in df.columns if df[column].isna().any()]

    # Achate a lista para evitar problemas de lista aninhada
    target_col = [item for sublist in target_col for item in (sublist if isinstance(sublist, list) else [sublist])]

    other_cols = df.columns.difference(target_col)
    
    # Dividir em dados com e sem NaNs
    df_no_nan = df.dropna(subset=target_col)
    df_with_nan = df[df[target_col].isna().any(axis=1)]
    
    # Preenchimento utilizando regressão linear
    # from sklearn.linear_model import LinearRegression
    
    for col in target_col:
        lr = LinearRegression()
        # Treinamento do modelo
        lr.fit(df_no_nan[other_cols], df_no_nan[col])
        
        # Preenchendo os valores faltantes
        df_with_nan[col] = lr.predict(df_with_nan[other_cols])
    
    # Combinar os DataFrames novamente
    df_filled = pd.concat([df_no_nan, df_with_nan]).sort_index()
    return df_filled


def fill_KNN(df):
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(n_neighbors=3)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed


def remove_NaN(df, threshold):
    # Remover colunas com mais de 50% de NaNs
    threshold = len(df) * threshold
    df_cleaned = df.dropna(axis=1, thresh=threshold)
    return df_cleaned

def fill_mean(df):
    # Preencher os valores restantes com a média da coluna
    df_filled = df.fillna(df.mean())
    return df_filled

def apply_pca(df, n_components):
  
    # Separar a primeira coluna (target)
    target = df.iloc[:, 0]
    features = df.iloc[:, 1:]

    # Aplicar PCA nas features
    pca = PCA(n_components=n_components)
    features_reduzidas = pca.fit_transform(features)

    # Converter para DataFrame
    df_reduzido = pd.DataFrame(features_reduzidas, index=df.index)
    
    # Adicionar a coluna target de volta como a primeira coluna
    df_final = pd.concat([target, df_reduzido], axis=1)

    # Renomear as colunas para manter a consistência com os nomes originais, exceto a coluna target
    df_final.columns = [df.columns[0]] + list(df.columns[1:n_components+1])
    
    return df_final

def plt_feature_importances(model, X_train, num_features):
    # Extrair importâncias das features
    importances = model.feature_importances_

    # Criar um DataFrame para facilitar a visualização
    feature_importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

    # Plotar as 10 variáveis mais importantes
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['feature'][:num_features], feature_importances_df['importance'][:num_features])
    plt.xlabel('Importance')
    plt.title('Top 10 Features Importance')
    plt.gca().invert_yaxis()  # Inverter o eixo y para exibir a maior importância no topo
    plt.show()
    return feature_importances_df

def top_features(num_features, df_feature_importances):
    top_10 = df_feature_importances[:num_features]
    return top_10

def plt_roc_curve(model, X_test, y_test):
    # Prever probabilidades
    y_prob = model.predict_proba(X_test)
    # Plotar as curvas ROC 
    plot_roc(y_test, y_prob, title='ROC Curves', figsize=(12, 5))

    plt.tight_layout()
    return print(plt.show())

def plt_precision_recall_curve_plot(model, X_test, y_test):
    
    # Previsões de probabilidade
    y_probas = model.predict_proba(X_test)
    
    # Plotar a curva Precision-Recall
    plot_precision_recall_curve(y_test, y_probas)
    return print(plt.show())

def plot_confusion_matrix_plot(model, X_test, y_test):
    # Previsões
    y_pred = model.predict(X_test)
    
    # Plotar a matriz de confusão
    plot_confusion_matrix(y_test, y_pred)
    return print(plt.show())

def saving_model(model, name):
    joblib.dump(model, name)
    return

def loading_model(file_name):
    
    return joblib.load(file_name)

def train_test(df):
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    return X_train, X_test, y_train, y_test

def fit_model_analisys(X_train, X_test, y_train, y_test):
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report, model

def test(df):
    X_test = df.iloc[:, 1:]
    y_test = df.iloc[:, 0]
    return X_test, y_test

def fit_model_analisys_test(model, X_test, y_test):

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report