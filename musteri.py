import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    dataset = pd.read_csv("titanic.csv")
    print(" Titanic veri seti yüklendi.")
except Exception as e:
    print(f" Veri seti yüklenemedi: {e}")
    raise SystemExit()

expected_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_')
selected_features = [col for col in expected_features if col in dataset.columns]
data = dataset[selected_features].fillna(dataset[selected_features].mean(numeric_only=True))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

inertia = []
silhouette_scores = []
ch_scores = []
db_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, labels))
    ch_scores.append(calinski_harabasz_score(scaled_data, labels))
    db_scores.append(davies_bouldin_score(scaled_data, labels))

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Inertia')
plt.xlabel('K')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_scores, marker='s', color='purple')
plt.title('Silhouette Score')
plt.xlabel('K')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(k_range, db_scores, marker='^', color='green')
plt.title('Davies-Bouldin Score')
plt.xlabel('K')
plt.grid(True)

plt.tight_layout()
plt.show()

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
dataset = dataset.loc[data.index]
dataset['Cluster'] = kmeans.fit_predict(scaled_data)

cluster_summary = dataset.groupby('Cluster')[selected_features].agg(['mean', 'std', 'min', 'max']).round(2)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
dataset['PCA1'] = pca_result[:, 0]
dataset['PCA2'] = pca_result[:, 1]

tsne_result = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(scaled_data)
dataset['TSNE1'] = tsne_result[:, 0]
dataset['TSNE2'] = tsne_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=dataset, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title('PCA Görselleştirme')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=dataset, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10')
plt.title('t-SNE Görselleştirme')
plt.grid(True)
plt.show()

for col in selected_features:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=dataset, x='Cluster', y=col, palette='pastel')
    plt.title(f'{col.upper()} vs Cluster')
    plt.grid(True)
    plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(cluster_summary['mean'], annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Her Kümenin Ortalama Değerleri")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Cluster', data=dataset, palette='Set2')
plt.title('Küme Dağılımı')
plt.xlabel('Küme')
plt.ylabel('Kişi Sayısı')
plt.grid(True)
plt.show()

corr = dataset[selected_features + ['Cluster']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()

if 'sex' in dataset.columns:
    grouped_sex = dataset.groupby(['sex', 'Cluster']).size().unstack()
    grouped_sex.plot(kind='bar', stacked=True, colormap='Accent', figsize=(8,5))
    plt.title("Cinsiyet ve Küme Dağılımı")
    plt.ylabel("Kişi Sayısı")
    plt.grid(True)
    plt.show()

if 'class' in dataset.columns:
    grouped_class = dataset.groupby(['class', 'Cluster']).size().unstack()
    grouped_class.plot(kind='bar', stacked=True, colormap='Paired', figsize=(8,5))
    plt.title("Sınıf ve Küme Dağılımı")
    plt.ylabel("Kişi Sayısı")
    plt.grid(True)
    plt.show()

if 'age' in dataset.columns and 'fare' in dataset.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=dataset, x='age', y='fare', hue='Cluster', palette='Dark2')
    plt.title('Yaş vs Üret - Küme Renkli')
    plt.grid(True)
    plt.show()

pd.set_option('display.max_columns', None)
print("Kümeleme Özeti:")
print(cluster_summary)

dataset.to_csv("clustered_titanic.csv", index=False)
print("\u2705 Sonuçlar 'clustered_titanic.csv' olarak kaydedildi.")
