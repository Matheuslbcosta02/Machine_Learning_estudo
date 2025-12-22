import xgboost as xgb
modelo = xgb.XGBClassifier(learning_rate =0.1 , max_depth = 10, n_estimators = 1000)

import pandas as pd
caminho_arquivo = 'train.gz'
df=pd.read_csv(caminho_arquivo,compression='gzip',nrows=300_000)
x = df.drop(['click','id','hour','device_id','device_ip'],axis=1).values
y = df['click'].values
treino_n = int(300000 * 0.9)
treino_x = x[:treino_n]
treino_y = y[:treino_n]
teste_x = x[treino_n:]
teste_y = y[treino_n:]

from sklearn.preprocessing import OneHotEncoder
codificar = OneHotEncoder(handle_unknown='ignore')

x_treino_codificado = codificar.fit_transform(treino_x)

x_teste_codificado = codificar.transform(teste_x)
modelo.fit(x_treino_codificado,treino_y)
pos_prob = modelo.predict_proba(x_teste_codificado)[:,1]
from sklearn.metrics import roc_auc_score
print(f'AUC da curva ROC no conjunto de teste usando GBT é {roc_auc_score(teste_y,pos_prob):.3f}')


