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
plt.title("Distribuição de Classes no Treino") #adicionei um titulo
plt.show()

partB.info()
plt.hist(partB['Class'])
plt.xlabel('Labels')
plt.ylabel('Freq')
plt.title("Distribuição de Classes no Teste") #titulo novamente para ficar organizado
plt.show()

y_train = partA["Class"]
X_train = partA.drop("Class", axis=1)

y_test = partB["Class"]
X_test = partB.drop("Class", axis=1)

k_range = np.arange(1, 20, 2)

best_score = 0
best_params = {}

for k in k_range:
    for weights in ["uniform", "distance"]:
        for metric in ["minkowski", "euclidean", "manhattan"]:
            for p in [1, 2]:
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, p=p)
                score = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy").mean()

                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "weights": weights, "metric": metric, "p": p}


clf = KNeighborsClassifier(n_neighbors=best_params["k"],
                           weights=best_params["weights"],
                           metric=best_params["metric"],
                           p=best_params["p"])


y_pred = cross_val_predict(clf, X_test, y_test, cv=10)
conf_matrix = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf_matrix.ravel()

print("TN:", tn)
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print()
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Precision:", precision_score(y_test, y_pred) * 100)
print("Recall:", recall_score(y_test, y_pred) * 100)
