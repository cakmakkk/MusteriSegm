# 📊 Müşteri Segmentasyonu Projesi (K-Means)
# 🔧 Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 📥 1. Veri Setini Yükleme (Seaborn veya CSV)
try:
    # Seaborn üzerinden hazır veri seti
    dataset = sns.load_dataset('titanic')
    print("✅ Seaborn ile Titanic veri seti başarıyla yüklendi.")
    # Titanic verisinden sayısal özellikleri seçiyoruz
    expected_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
except Exception as e:
    print(f"🚨 Veri seti yüklenemedi: {e}")
    raise SystemExit()

# 📝 2. Veri Setinin İlk 5 Satırını Görüntüleyin
print("Veri Seti İlk 5 Satır:")
print(dataset.head())

# 🧹 3. Sütun İsimlerini Düzenleme
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_')

# 🔍 4. Beklenen Sütunları Kontrol Edin
selected_features = [col for col in expected_features if col in dataset.columns]

# 🛑 Eksik Sütunları Bildirin
missing_columns = set(expected_features) - set(selected_features)
if missing_columns:
    print(f"⚠️ Eksik sütunlar bulunamadı: {missing_columns}")
    print(f"Yalnızca şu sütunlarla devam ediliyor: {selected_features}")

# ✅ 5. Seçilen Sütunlarla Veri Hazırlığı
if not selected_features:
    raise ValueError("Veri setinde kullanılabilir sütun bulunamadı. Lütfen doğru veri setini kullanın.")

data = dataset[selected_features]

# 🧼 6. Eksik Değerleri Kontrol Edin ve Gerekirse Temizleyin
print("\nEksik Değerler:")
print(data.isnull().sum())
data = data.dropna()

# ⚙️ 7. Veriyi Ölçeklendirme
if data.empty:
    raise ValueError("Veri seti boş! Lütfen geçerli bir veri kaynağı kullanın.")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print("✅ Veriler başarıyla ölçeklendirildi.")

# 📉 8. Optimum Küme Sayısını Belirleme (Elbow ve Silhouette)
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# 📊 Elbow Yöntemi Grafiği
plt.figure(figsize=(7,4))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Yöntemi - Optimum Küme Sayısı')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('Inertia (Hata Kareleri Toplamı)')
plt.grid(True)
plt.show()

# 📊 Silhouette Skoru Grafiği
plt.figure(figsize=(7,4))
plt.plot(k_range, silhouette_scores, marker='s', linestyle='--', color='purple')
plt.title('Silhouette Skoru')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('Silhouette Skoru')
plt.grid(True)
plt.show()

# 📍 9. K-Means Modeli ile Kümeleme
optimal_clusters = 4  # Elbow ve Silhouette sonucuna göre
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
dataset['Cluster'] = kmeans.fit_predict(scaled_data)

# 📈 10. Kümeleme Sonuçlarını Analiz Etme
print("\nKümeleme Sonuçları (Ortalama Değerler):")
cluster_summary = dataset.groupby('Cluster').mean()
print(cluster_summary)

# 🟡 11. Sonuçları Görselleştirme (Scatterplot)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=dataset[selected_features[0]], 
    y=dataset[selected_features[1]],
    hue=dataset['Cluster'],
    palette='viridis',
    legend='full'
)
plt.title(f'Müşteri Segmentasyonu ({selected_features[0]} vs {selected_features[1]})')
plt.xlabel(selected_features[0].capitalize().replace('_', ' '))
plt.ylabel(selected_features[1].capitalize().replace('_', ' '))
plt.grid(True)
plt.show()

# 📊 12. Küme Başına Müşteri Sayısı
plt.figure(figsize=(5,4))
sns.countplot(x='Cluster', data=dataset, palette='Set2')
plt.title('Her Kümede Müşteri Sayısı')
plt.xlabel('Küme')
plt.ylabel('Müşteri Sayısı')
plt.grid(True)
plt.show()

# 📝 13. Küme Özelliklerinin Isı Haritası
plt.figure(figsize=(8,5))
sns.heatmap(cluster_summary, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Küme Bazında Ortalama Değerler')
plt.show()
