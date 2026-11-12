import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_random_forest(X_train, y_train):

        pca_big = PCA(n_components=100)
        pca_big.fit(X_train)

        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('Зависимость доли сохраненной информации от количества компонент PCA')
        sns.set(style='whitegrid')
        plt.plot(np.cumsum(pca_big.explained_variance_ratio_), label='Накопленная объяснённая дисперсия')
        plt.xlabel('Число компонент')
        plt.ylabel('Объяснённая дисперсия')
        plt.axvline(linewidth=4, color='r', linestyle='--', x=30, ymin=0, ymax=1, label='Число компонент, достаточное для сохранения 95% информации при PCA (30)')
        plt.legend()
        plt.savefig('results/cumulative_variance_ratio(rf).png', dpi=300, bbox_inches='tight')
        plt.close()

        pca = PCA(n_components=30)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)

        rfc = RandomForestClassifier()
        rfc.fit(X_train_pca, y_train)

        return rfc, pca

def test_random_forest(rfc, pca, X_test, y_test):
    X_test_pca = pca.transform(X_test)
    result = rfc.predict(X_test_pca)
    print(classification_report(y_test, result))
    cm = confusion_matrix(y_test, result)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    tick_labels = ['I','II','III','IV','V','VI']

    plt.title('Матрица ошибок')
    hmap = sns.heatmap(cm_normalized,
                         square=True,
                         annot=True,
                         cbar=False,
                         cmap='GnBu',
                         xticklabels=tick_labels,
                         yticklabels=tick_labels)
    plt.ylabel('Истинные категории')
    plt.xlabel('Предсказанные категории')
    plt.savefig('results/confusion_matrix(rf).png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.title('Сравнение истинных и предсказанных значений')
    x_pos = np.array([0, 1, 2, 3, 4, 5])
    y_actual = [(y_test == 'I').sum(), (y_test == 'II').sum(), (y_test == 'III').sum(),
                (y_test == 'IV').sum(), (y_test == 'V').sum(), (y_test == 'VI').sum()]
    y_predict = [(result == 'I').sum(), (result == 'II').sum(), (result == 'III').sum(),
                 (result == 'IV').sum(), (result == 'V').sum(), (result == 'VI').sum()]
    width = 0.5
    plt.bar(x_pos - width/2, y_actual, width, edgecolor='black', label='Истинные')
    plt.bar(x_pos + width/2, y_predict, width, edgecolor='black', label='Предсказанные')
    x_labels = ['I','II','III','IV','V','VI']
    plt.xticks(x_pos, x_labels)
    plt.ylabel('Категории')
    plt.xlabel('Количество')
    plt.legend()
    plt.savefig('results/сomparison_true_predicted(rf).png', dpi=300, bbox_inches='tight')
    plt.close()
    print("success")

def create_random_forest(embeddings_name):
    embeddings_df = pd.read_excel(embeddings_name)
    category = embeddings_df['Category']
    embeddings = embeddings_df.drop('Category', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, category, test_size=0.1, random_state=2020,
                                                        stratify=category)

    print("training random forest...")
    rfc, pca = train_random_forest(X_train, y_train)
    print("success")
    print("classifying data...")
    test_random_forest(rfc, pca, X_test, y_test)
    print("success")

    joblib.dump(rfc, 'random_forest.joblib')
    joblib.dump(pca, 'pca_model.joblib')
    return pca, rfc