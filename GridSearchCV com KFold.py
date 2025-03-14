import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/content/PDFMalware2022_pp.csv", dtype={"Class": int})

# Extrair features e labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Escalar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KNN
knn = KNeighborsClassifier()

kf = KFold(n_splits=5, shuffle=True)

# Parâmetros de otimização com o GridSearch
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'cityblock']
}

# GridSearch com validação cruzada
gs = GridSearchCV(knn, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

print(f'Parâmetros: {gs.best_params_}')
print(f'Acurácia treino: {gs.best_score_:.5f}')



# Para ter certeza, utilizar os parâmetros pelo Grid e rodar com KFoldCross
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')

scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')

print(f'Scores: {scores}')
print(f'Acurácia média: {np.mean(scores):.5f}')
