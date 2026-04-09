import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("casas_california.csv")

# CRIA CATEGORIAS E BINS

df['categoria_preco'], bins = pd.qcut(df['valor_mediano_imoveis'], q=3, 
labels=['Baixo', 'Médio', 'Alto'], retbins=True )

# PLOTAGEM

plt.figure(figsize=(10, 5))

sns.histplot(df['valor_mediano_imoveis'], bins=50, kde=True)

# LINHAS DE CORTE
plt.axvline(bins[1], linestyle = '--')
plt.axvline(bins[2], linestyle = '--')

# TEXTO (POSIÇÃO AJUSTAVEL)
y_max = plt.gca() .get_ylim()[1]

plt.text(bins[0], y_max * 0.8 , "Baixo")
plt.text(bins[1], y_max *0.8, "Médio")
plt.text(bins[2], y_max *0.8, "Alto")

# LABELS

plt.title("categorias de preço dos imóveis QCUT")
plt.xlabel("Valor Mediano dos Imóveis")
plt.ylabel("Frequência")
plt.show()


## TRANSFORMANDO VARIÁVEIS NÃO NUMÉRICAS EM NUMÉRICAS

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le_target = LabelEncoder()
df['categoria_preco'] = le_target.fit_transform(df['categoria_preco'])

# convertendo ocean_proximity em numérico
le = LabelEncoder()
df['proximidade_oceano'] = le.fit_transform(df['proximidade_oceano'])

print(df)



## Seleção das entradas e saída Preparação dos dados de treino e teste

# FEATURES (X) e TARGET (y)
x = df.drop(['valor_mediano_imoveis', 'categoria_preco'], axis=1)
y = df['categoria_preco']


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_treino.shape)
print(x_teste.shape)
print(y_treino.shape)
print(y_teste.shape)


from xgboost import XGBClassifier
model = XGBClassifier(
n_estimators=100,
max_depth=4,
learning_rate=0.1,
random_state=42)
model.fit(x_treino, y_treino)


y_pred = model.predict(x_teste)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_teste, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_teste, y_pred))



import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_teste, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d',
cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()