import numpy as np
import pandas as pd
caminho_arquivo = 'heart+disease/processed.cleveland.data'

df = pd.read_csv(caminho_arquivo,header=None,sep=',', engine='python')
df.columns = ['idade','sexo','tipo_de_dor_no_peito','pressão_arterial_repouso','colesterol','acucar_sangue','eletrocardiograma','freq_cardiaca_max','angina_no_exerc','depressao_ST_no_ecg','inclinacao_ST_no_ECG','vasos_vistos','tipo_talessemia','TARGET_DOENCA_CARDIACA']
df['doenca_cardiaca_binaria'] = (df['TARGET_DOENCA_CARDIACA']>0).astype(int)


def validar_valor(x):
    try:
        valor = float(x)
        if valor < 0 :
            return float('nan')
        return valor
    except:
        return float('nan')
    
df_tratado = df.applymap(validar_valor)
print(df_tratado)
df_tratado.dropna()
df_tratado.to_csv('dados_cardiacos_tratados.csv', index=False)


X = df_tratado.drop(columns=['TARGET_DOENCA_CARDIACA','doenca_cardiaca_binaria'])
Y = df_tratado['doenca_cardiaca_binaria']
X = X.fillna(X.mean())

from sklearn.model_selection import train_test_split

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2,random_state=42)

from sklearn.naive_bayes import GaussianNB

meu_classificador = GaussianNB()
meu_classificador.fit(X_treino, Y_treino)

probabilidade_da_predicao = meu_classificador.predict_proba(X_teste)
predicao_cada_paciente_amostra = meu_classificador.predict(X_teste)

x_teste_8 = X_teste.iloc[:8]
probabilidade_da_predicao_8_pacientes = probabilidade_da_predicao[:8]
predicao_8_pacientes = predicao_cada_paciente_amostra[:8]

df_resultado_8_pacientes = x_teste_8.copy()
df_resultado_8_pacientes['probabilidade 1'] = probabilidade_da_predicao_8_pacientes[:,1]
df_resultado_8_pacientes['probabilidade 0'] = probabilidade_da_predicao_8_pacientes[:,0]
df_resultado_8_pacientes['predicao da doenca cardiaca'] = predicao_8_pacientes 

df_resultado_8_pacientes.to_csv('resultado_8_pacientes.csv', index=False)

tabela_ajustada_theu = df_resultado_8_pacientes.to_string(index=False)
with open('resultado_8_pacientes.txt', mode='w',encoding='utf-8') as arquivo:
    arquivo.write(tabela_ajustada_theu)


acuracia = meu_classificador.score(X_teste, Y_teste)
print(f"A acurácia do meu modelo de predição para doença coronariana é {acuracia*100:.2f}%")

from sklearn.metrics import confusion_matrix
matriz_confusao = confusion_matrix(Y_teste,predicao_cada_paciente_amostra,labels=[0,1])
print(matriz_confusao)

from sklearn.metrics import classification_report
reporte_geral = classification_report(Y_teste,predicao_cada_paciente_amostra)
print(reporte_geral)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

probabilidades_positivas = probabilidade_da_predicao[:,1]
eixo_x_falso_positivo, eixo_y_verdadeiro_Positivo, thresholds_cortes_usados = roc_curve(Y_teste,probabilidades_positivas)

auc = roc_auc_score(Y_teste,probabilidades_positivas)
print(f"AUC: {auc}")

plt.figure()
plt.plot(eixo_x_falso_positivo,eixo_y_verdadeiro_Positivo, color = 'orange', lw=2, label=f'curva ROC(AUC = {auc})')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle=':')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('Falso Positivo FPR')
plt.ylabel("Verdadeiro positivo TPR")
plt.title("Curva ROC")
plt.legend(loc='lower right')
plt.show()


from sklearn.model_selection import StratifiedKFold
k = 5
k_fold = StratifiedKFold(n_splits=k, shuffle=True,random_state=42)
opcoes_suavizar = [1e-9,1e-8,1e-7,1e-6]
gravar_auc ={}

for indices_treino, indices_teste in k_fold.split(X,Y):
    X_treino_k = X.iloc[indices_treino]
    X_teste_k = X.iloc[indices_teste]
    Y_treino_k = Y.iloc[indices_treino]
    Y_teste_k = Y.iloc[indices_teste]

    for vs in opcoes_suavizar:
        if vs not in gravar_auc:
            gravar_auc[vs] = 0.0
        clf = GaussianNB(var_smoothing=vs)
        clf.fit(X_treino_k,Y_treino_k)
        prob_pos = clf.predict_proba(X_teste_k)[:,1]
        auc = roc_auc_score(Y_teste_k,prob_pos)
        gravar_auc[vs] += auc

for vs,auc in gravar_auc.items():
    print(f'opção de suavizar = {vs} -> AUC média = {auc/k:.4f}')

clf_final = GaussianNB(var_smoothing=1e-7)
clf_final.fit(X_treino,Y_treino)
prob_pos = clf_final.predict_proba(X_teste)[:,1]
print("AUC com melhor modelo:", roc_auc_score(Y_teste,prob_pos))

