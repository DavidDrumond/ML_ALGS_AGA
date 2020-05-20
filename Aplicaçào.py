import pickle
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from yellowbrick.model_selection import RFECV
from yellowbrick.datasets import load_energy
from yellowbrick.model_selection import ValidationCurve
from yellowbrick.regressor import PredictionError
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pandas as pd 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import re 
import pickle 
import os
from pyod.models.iforest import IForest
from pyod.utils.example import visualize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib._color_data as mcd
import seaborn as sns
import matplotlib
from statsmodels.tsa.seasonal import seasonal_decompose
import time
from datetime import datetime
import scipy
import rpy2.robjects as ro
import plotly.graph_objects as go
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import Rank2D
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer






# FUNÇOES NECESSARIAS -- 
# CRIAÇÃO DAS VARIÁVEIS TOTAIS
def valores_totais(df):
    COLUNAS_OURO = [i for i in df.columns if  re.findall('-Au', i) ]
    COLUNAS_ENXOFRE = [i for i in df.columns if  re.findall('-S', i) ]
    COLUNAS_MASSA = [i for i in df.columns if  re.findall('Massa', i) ]
    SHAPE = df[COLUNAS_OURO[0]].shape[0]
    totais_ouro, totais_massa, totais_enxofre =  np.zeros(SHAPE), np.zeros(SHAPE),np.zeros(SHAPE)
    for i, j, k in zip(COLUNAS_OURO, COLUNAS_MASSA, COLUNAS_ENXOFRE):
        totais_ouro += np.multiply(df[i].values,df[j].values)
        totais_enxofre += np.multiply(df[k].values,df[j].values)
        totais_massa += df[j].values
    RECUPERACAO_OURO = [i/float(j) if j!= 0.000 else 0.00 for i, j in zip (totais_ouro, totais_massa)]
    RECUPERACAO_ENXOFRE = [i/float(j) if j!= 0.000 else 0.00 for i, j in zip (totais_enxofre, totais_massa)]
    df['Total gold (g/t)'] = np.array(RECUPERACAO_OURO)
    df['Total sulfur (%)'] = np.array(RECUPERACAO_ENXOFRE)
    df['Total mass (t)'] = totais_massa
    return df



# IMPORTAR OS DADOS 
def importar_dados(endereco_quimicas, endereco_operacionais, endereco_output, janela_temporal, tempo_residencia, TEMPO_PLANEJAMENTO): 
    DATA_FRAME_QUIMICAS = pd.read_excel(endereco_quimicas, index_col='Data')
    DATA_FRAME_OPERACIONAIS = pd.read_excel(endereco_operacionais, index_col='Data')
    DATA_FRAME_OUTPUT = pd.read_excel(endereco_output, index_col='Data')
    ADICAO_VARIAVEIS_TOTAIS  = valores_totais(DATA_FRAME_QUIMICAS)
    ADICAO_VARIAVEIS_TOTAIS  = pd.concat([ADICAO_VARIAVEIS_TOTAIS ,DATA_FRAME_OPERACIONAIS ], join='inner', axis=1)
    
    ADICAO_VARIAVEIS_TOTAIS = ADICAO_VARIAVEIS_TOTAIS.loc[:,~ADICAO_VARIAVEIS_TOTAIS.columns.duplicated()]              # remova colunas duplicadas acidentalmente
    ADICAO_VARIAVEIS_TOTAIS.index = pd.to_datetime(ADICAO_VARIAVEIS_TOTAIS.index)
    ADICAO_VARIAVEIS_TOTAIS = ADICAO_VARIAVEIS_TOTAIS.resample(janela_temporal).mean()                                   # realizar medidas no suporte temporal desejado
    ADICAO_VARIAVEIS_TOTAIS.drop(ADICAO_VARIAVEIS_TOTAIS.head(tempo_residencia).index,inplace=True) 
    ADICAO_VARIAVEIS_TOTAIS = ADICAO_VARIAVEIS_TOTAIS.dropna()

    DATA_FRAME_OUTPUT = DATA_FRAME_OUTPUT.loc[:,~DATA_FRAME_OUTPUT.columns.duplicated()]              # remova colunas duplicadas acidentalmente
    DATA_FRAME_OUTPUT.index = pd.to_datetime(DATA_FRAME_OUTPUT.index)
    DATA_FRAME_OUTPUT = DATA_FRAME_OUTPUT.resample(janela_temporal).mean()               # realizar medidas no suporte temporal desejado
    DATA_FRAME_OUTPUT.drop(DATA_FRAME_OUTPUT.tail(tempo_residencia).index,inplace=True) 
    DATA_FRAME_OUTPUT = DATA_FRAME_OUTPUT.dropna()

    DATA_FRAME_OUTPUT = DATA_FRAME_OUTPUT[DATA_FRAME_OUTPUT['Rec Au (%)']> 87]
    DECOMPOSICAO = seasonal_decompose(DATA_FRAME_OUTPUT.values, model='additive', period=TEMPO_PLANEJAMENTO)
    plt.hist(DECOMPOSICAO.resid)
    plt.show()

    #TRANSFORMADA = FunctionTransformer(np.log1p).transform(ADICAO_VARIAVEIS_TOTAIS)
    COMPONENTES_PRINCIPAIS = PCA(n_components=5, svd_solver='full').fit_transform(ADICAO_VARIAVEIS_TOTAIS)
    print(COMPONENTES_PRINCIPAIS.shape)
    return COMPONENTES_PRINCIPAIS, np.nanstd(DECOMPOSICAO.resid), DECOMPOSICAO.seasonal, ADICAO_VARIAVEIS_TOTAIS.index


def estimar_valores(endereco_quimicas, endereco_operacionais, endereco_output, janela_temporal, tempo_residencia, TEMPO_PLANEJAMENTO):
    pca, DESVPAD_RUIDO, SAZONAL, data = importar_dados(endereco_quimicas, endereco_operacionais, endereco_output, janela_temporal, tempo_residencia, TEMPO_PLANEJAMENTO)
    print(DESVPAD_RUIDO)
    filename =  'D:/Drive/AGA - GEOMET/AVALIAÇÃO/5 - APLICAÇÃO/modelo.sav'
    model = load(filename)
    VALORES_PREDITOS = model.predict(pca)
    VALORES_PREDITOS = VALORES_PREDITOS + SAZONAL
    LIMITE_SUPERIOR = VALORES_PREDITOS + 2*DESVPAD_RUIDO
    LIMITE_INFERIOR = VALORES_PREDITOS - 2*DESVPAD_RUIDO
    TERC_QUARTIL = VALORES_PREDITOS + DESVPAD_RUIDO
    PRIM_QUARTIL = VALORES_PREDITOS - DESVPAD_RUIDO
    DATAFRAME_SAIDA = pd.DataFrame(np.array([data, VALORES_PREDITOS, LIMITE_SUPERIOR, LIMITE_INFERIOR, TERC_QUARTIL, PRIM_QUARTIL]).T,
                                    columns = ['data', 'VALOR ESPERADO', 'LIMITE SUPERIOR', 'LIMITE INFERIOR', 'TERCEIRO QUARTIL', 'PRIMEIRO QUARTIL'])
    DATAFRAME_SAIDA.to_excel('D:/Drive/AGA - GEOMET/AVALIAÇÃO/5 - APLICAÇÃO/RESULTADOS.xlsx')


def __init__():
    JANELA_TEMPORAL = '7d'
    TEMPO_RESIDENCIA = 2
    TEMPO_PLANEJAMENTO = 30
    endereco_quimicas = 'D:/Drive/AGA - GEOMET/AVALIAÇÃO/5 - APLICAÇÃO/Variáveis_químicas.xlsx'
    endereco_operacionais = 'D:/Drive/AGA - GEOMET/AVALIAÇÃO/5 - APLICAÇÃO/Condicionamento.xlsx'
    endereco_output = 'D:/Drive/AGA - GEOMET/AVALIAÇÃO/5 - APLICAÇÃO/RECUPERACOES.xlsx'
    estimar_valores(endereco_quimicas, endereco_operacionais, endereco_output, JANELA_TEMPORAL,TEMPO_RESIDENCIA, TEMPO_PLANEJAMENTO)

__init__()