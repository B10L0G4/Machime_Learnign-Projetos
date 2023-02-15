import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 


df= pd.read_csv('./arquivos/temperature.csv')

#extração de x e y que serão variareis preditoras que iremos usar e as variaveis target que queremos  usar prever - será usado apenas a temperatura para esses projeto. 

# a classe LabelEncoder é usada para transformar os valores de uma coluna em valores numericos


x, y = df[['temperatura']].values,df[['classification']].values

#print("x:\n",x)
#print("y:\n",y)

#print(df.head)
le = LabelEncoder()
y = le.fit_transform(y.ravel()) #rave() transforma o array em uma linha, fit_transform( transforma os valores em numericos )

# print("y:\n",y)

#linear regression é um modelo de regressão linear que é usado para prever valores continuos 
#a classe LogisticRegression é usada para prever valores discretos, ou seja, valores que não são continuos 

# classificação 

clf = LogisticRegression() # criando o modelo
clf.fit(x, y) #fit é usado para treinar o modelo 


#gerando 100 valores de temperatura para serem usados para prever a classificação
x_test =np.linspace(start=0, stop=45.,num=100).reshape(-1,1) #reshape(-1,1)transforma o array em uma coluna 
y_pred = clf.predict(x_test) #predict é usado para prever os valores

y_pred = le.inverse_transform(y_pred)
#print(y_pred)

#output 
#output = {'new_temp': x_test.ravel(),             'new_class':y_pred.ravel()} #ravel()transforma o array em uma linha , o output é um dicionario que será usado para criar um dataframe

output = {'new_temp': x_test.ravel(),'new_class': y_pred.ravel()}

output = pd.DataFrame(output) #criando um dataframe com o ourput

output.head() #mostra os 5 primeiros valores do dataframe
#output.tail() #mostra os 5 ultimos valores do dataframe
#output.info() #mostra as informações do dataframe
