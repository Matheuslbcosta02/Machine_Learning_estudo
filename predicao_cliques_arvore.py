import pandas as pd
caminho_arquivo = "train.gz"

df = pd.read_csv(caminho_arquivo,compression='gzip',nrows=300_000)
print("linhas(amostras), colunas(features)",df.shape)
print("5 primeiras linhas por padrão:")
print()
print(df.head())

x = df.drop(['click','id','hour','device_id','device_ip'],axis=1).values
y = df['click'].values
print(x.shape)

treino_n = int(300000 * 0.9)
treino_x = x[:treino_n]
treino_y = y[:treino_n]
teste_x = x[treino_n:]
teste_y = y[treino_n:]

from sklearn.preprocessing import OneHotEncoder
codificar = OneHotEncoder(handle_unknown='ignore')

x_treino_codificado = codificar.fit_transform(treino_x)

x_teste_codificado = codificar.transform(teste_x)


from sklearn.tree import DecisionTreeClassifier
parametros_profundidade_arvore = {'max_depth':[3,10,None]}
minha_arvore = DecisionTreeClassifier(criterion="gini",min_samples_split=30)

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(minha_arvore,parametros_profundidade_arvore,n_jobs=-1,cv=3,scoring='roc_auc')
grid_search.fit(x_treino_codificado,treino_y)
print(grid_search.best_params_)
melhor_decisao_arvore = grid_search.best_estimator_
pos_prob = melhor_decisao_arvore.predict_proba(x_teste_codificado)[:,1]

from sklearn.metrics import roc_auc_score
print(f"O ROC AUC no conjunto de teste é {roc_auc_score(teste_y,pos_prob):.3f}")

from sklearn.ensemble import RandomForestClassifier
meu_conjunto_arvores = RandomForestClassifier(n_estimators=100,criterion='gini',min_samples_split=30,n_jobs=-1)
grid_search2 = GridSearchCV(meu_conjunto_arvores, parametros_profundidade_arvore,n_jobs=-1,cv=3, scoring='roc_auc')
grid_search2.fit(x_treino_codificado,treino_y)
print(grid_search2.best_params_)
melhor_conjunto_arvores = grid_search2.best_estimator_
pos_prob2 = melhor_conjunto_arvores.predict_proba(x_teste_codificado)[:,1]
print(f'A melhor ROC AUC no teste usando floresta aleatória é: {roc_auc_score(teste_y,pos_prob2):.3f}')


