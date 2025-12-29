from sklearn.linear_model import SGDClassifier

import pandas as pd

caminho_arquivo = 'train.gz'
linhas_lidas = 100_000
df = pd.read_csv(caminho_arquivo, compression='gzip',nrows=linhas_lidas)
x = df.drop(['click','id','hour','device_id','device_ip'],axis=1).values
y = df['click'].values

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.2, random_state=42)


from sklearn.preprocessing import OneHotEncoder

codificar = OneHotEncoder(handle_unknown='ignore')
x_treino_codificado = codificar.fit_transform(x_treino)
x_teste_codificado = codificar.transform(x_teste)

meu_SGD = SGDClassifier(loss='log_loss',penalty=None, fit_intercept=True, max_iter=20,learning_rate='constant',eta0=0.01)
meu_SGD.fit(x_treino_codificado.toarray(), y_treino)
pred = meu_SGD.predict_proba(x_teste_codificado.toarray())[:,1]

from sklearn.metrics import roc_auc_score

print(f"Amostras de treinamento {linhas_lidas}, AUC no conjunto de teste: {roc_auc_score(y_teste,pred):.3f}")
