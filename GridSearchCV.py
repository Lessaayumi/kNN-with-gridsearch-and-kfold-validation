import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/content/PDFMalware2022_pp.csv", dtype={"Class": int})

df.head()

df.describe()

df.info()

partA, partB = train_test_split(df, test_size=0.9)

partA.info()
plt.hist(partA['Class'])
plt.xlabel('Labels')
plt.ylabel('Freq')
plt.show()

partB.info()
plt.hist(partB['Class'])
plt.xlabel('Labels')
plt.ylabel('Freq')
plt.show()

y = partA["Class"]
X = partA.drop("Class", axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

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

best_params = gs.best_params_
best_k = best_params['n_neighbors']
best_weights = best_params['weights']
best_metric = best_params['metric']

scores = []
for k in np.arange(1, 20, 2):
    knn = KNeighborsClassifier(n_neighbors=k, weights=best_weights, metric=best_metric)
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy").mean()
    scores.append(score)

best_k_index = scores.index(max(scores))

plt.plot(np.arange(1, 20, 2), scores)
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.axvline(np.arange(1, 20, 2)[best_k_index], color="red", linestyle="dashed")
plt.xticks(np.arange(1, 20, 2))
plt.show()

y = partB["Class"]
X = partB.drop("Class", axis=1)

clf1 = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, metric=best_metric)
clf1_pred = cross_val_predict(clf1, X, y, cv=10)
conf_clf1 = confusion_matrix(y, clf1_pred)

tn = conf_clf1[0, 0]
tp = conf_clf1[1, 1]
fp = conf_clf1[0, 1]
fn = conf_clf1[1, 0]

print("TN:", tn)
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print()
print("Accuracy:", accuracy_score(y, clf1_pred) * 100)
print("Precision:", precision_score(y, clf1_pred) * 100)
print("Recall:", recall_score(y, clf1_pred) * 100)