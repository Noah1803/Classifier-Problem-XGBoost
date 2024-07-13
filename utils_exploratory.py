##Importando bibliotecas para construir funções 
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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


def df_csv(csv):
    nomedf = pd.read_csv(csv)
    return nomedf 


def classe_boll_1(df):
    df['class'] = df['class'].replace('pos', 1)
    return df

def classe_boll_0(df):
    df['class'] = df['class'].replace('neg', 0)
    return df

def subplots_hist(df):
    num_cols = len(df.columns[1:])
    rows = int(np.ceil(num_cols / 3)) 
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for idx, col in enumerate(df.columns[1:]): 
        axes[idx].hist(df[col], bins=20, edgecolor='black')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequência')
        axes[idx].set_title(f'Histograma da coluna {col}')

    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return print(plt.show())


def filter_columns_corr(df, corr):
    # Calcular a matriz de correlação
    correlation_matrix = df.corr().abs()

    # Criar uma máscara para a matriz de correlação superior
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype('bool')
    )

    # Encontrar colunas com correlação superior a 50%
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr)]

    # Salvar as colunas identificadas para deleção
    with open('colunas_para_deletar.txt', 'w') as file:
        for column in to_drop:
            file.write(f"{column}\n")

    # Deletar as colunas do dataframe
    df_reduced = df.drop(columns=to_drop)

    # Opcional: Salvar o dataframe reduzido
    # df_reduced.to_csv('df_reduced.csv', index=False)

    # Exibir n colunas que serão deletadas
    print(len(to_drop))

    return df_reduced

def qtd_NaN(df):
    is_na = df.isna().sum()
    is_na = pd.DataFrame(is_na)
    is_na.columns = ['NaN Sum']
    return is_na

def col_3seq_na(df):
    vetor_nomes = []
    for col in df.columns:
        for i in df[col].index:
            
            if pd.isna(df[col][i]) and pd.isna(df[col][i+1]) and pd.isna(df[col][i+2]):
                vetor_nomes.append(col)
                if col in vetor_nomes:
                    break
    return vetor_nomes,len(vetor_nomes)


def count_classes(df):
    num_pos = df[df['class'] == 1].shape[0]
    num_neg = df[df['class'] == 0].shape[0]
    print({"num_pos" : num_pos, "num_neg" : num_neg})
    return num_pos,num_neg







