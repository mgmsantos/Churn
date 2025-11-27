# %%

import pandas as pd

# Analytic Base Table Churn
df = pd.read_csv("abt_churn.csv")
df.head()
df.shape

# %%

# Out of Time - Sample de teste futuro
oot = df[df['dtRef'] == df['dtRef'].max()].copy()

oot

# %%

# Sample de treinamento
df_training = df[df['dtRef'] < df['dtRef']. max()].copy()
df_training.shape

# %%

# Selecionar quais features serão interessantes
features = df_training.columns[2:-1]

# Selecionar a variável target
target = 'flagChurn'

x, y = df_training[features], df_training[target]

# %%

from sklearn import model_selection

x_training, x_test, y_training, t_test = model_selection.train_test_split(x, y,
                                                                          random_state = 42,
                                                                          test_size = 0.2,
                                                                          stratify = y
                                                                          )


# %%

# EDA

# Verificar dados faltantes (missing) na base de treino
x_training.isna().sum()

# %%

# Análise bivariada: descobrir quais features contribuem para previsão do target

df_analise = x_training.copy()
df_analise[target] = y_training
summary = df_analise.groupby(by = target).agg(['mean', 'median']).T
summary

# %%

from sklearn import tree
import matplotlib.pyplot as plt

# Fazendo uma árvore de decisão para verificar a importancia de cada feature

arvore = tree.DecisionTreeClassifier(random_state = 42)
arvore.fit(x_training, y_training)

# %%

# Ordenando a importancia de cada feature
feature_importance = (pd.Series(arvore.feature_importances_, 
                               index = x_training.columns)
                               .sort_values(ascending = False)
                               .reset_index()
                                )
feature_importance.columns = ['index', 'importance']
feature_importance

# %%

# Acumulando a importance das features
feature_importance['acumulado'] = feature_importance['importance'].cumsum()
feature_importance

# %%

# Selecionar features que com importance > 1%
feature_importance[feature_importance['importance'] > 0.01]

# ou selecionar a importance acumulada até 95%
feature_importance[feature_importance['acumulado'] < 0.96]

# %%
