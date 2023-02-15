import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 


df= pd.read_csv('./arquivos/temperature.csv')

#extração de x e y que serão variareis preditoras que iremos usar e as variaveis target que queremos  usar prever - será usado apenas a temperatura para esses projeto. 

# a classe LabelEncoder é usada para transformar os valores de uma coluna em valores numericos


x, y = df[['temperatura']],df[['classification']].values

print("x:\n",x)
print("y:\n",y)

le = LabelEncoder()
y = le.fit_transform(y.ravel()) #rave() transforma o array em uma linha, fit_transform( transforma os valores em numericos )

print("y:\n",y)

#linear regression é um modelo de regressão linear que é usado para prever valores continuos 
#a classe LogisticRegression é usada para prever valores discretos, ou seja, valores que não são continuos 

# classificação 

clf = LogisticRegression() # criando o modelo
clf.fit(x,y) #fit é usado para treinar o modelo 

