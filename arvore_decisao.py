from sklearn.tree import DecisionTreeClassifier
minha_arvore = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
x_texto =[['carro'],['moda'],['esporte'],['tecnologia'],['educacional']]
y_treino = [1,0,1,1,0]

from sklearn.preprocessing import OneHotEncoder
codificar = OneHotEncoder(sparse_output=False)
x_treino = codificar.fit_transform(x_texto)
feature_names = codificar.get_feature_names_out()
minha_arvore.fit(x_treino, y_treino)

from sklearn.tree import export_graphviz
export_graphviz(minha_arvore, out_file = 'tree.dot', feature_names = feature_names,impurity=False,filled = True, class_names=['0','1'])
#converter dot para png --> dot -t png [nome.dot] -o [nome.png]
#instalar o dot

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plot_tree(minha_arvore,feature_names=feature_names,class_names=['0','1'],filled=True,rounded=True,fontsize=10,proportion=False)
plt.show()
