<div align="center">

# 🌸 Klasifikasi Jenis Bunga Berbasis Convolutional Neural Network (CNN)
**Studi Kasus: Dataset Oxford 102 Flower Menggunakan Arsitektur MobileNetV2**

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Transfer_Learning-red?style=for-the-badge&logo=keras&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-GPU_Enabled-yellow?style=for-the-badge&logo=googlecolab&logoColor=white)

</div>

---

## 👤 Identitas Penulis
**Universitas Muhammadiyah Sukabumi**
* **Nama :** Maulana Ifnu Syafi
* **NIM :** 2230511064

---

## 1. Latar Belakang
Indonesia dan dunia memiliki keanekaragaman flora yang sangat kaya. Identifikasi jenis bunga sangat membantu di berbagai bidang, mulai dari konservasi botani hingga budidaya tanaman hias. Namun, proses identifikasi manual seringkali menghadapi kendala, terutama dengan banyaknya spesies yang ada. Bagi orang awam, banyak bunga yang memiliki kemiripan bentuk dan warna (visual), sehingga rentan terjadi kesalahan identifikasi (*human error*).

Dengan berkembangnya teknologi kecerdasan buatan, khususnya *Deep Learning*, masalah tersebut dapat diatasi secara otomatis. *Convolutional Neural Network* (CNN) terbukti menjadi metode yang sangat efektif untuk pengolahan citra digital. Kemampuannya untuk mengekstraksi fitur visual—mulai dari garis sederhana hingga pola kompleks—menjadikannya arsitektur yang ideal untuk klasifikasi objek.

Proyek ini menggunakan dataset publik **Oxford 102 Flower** serta arsitektur **MobileNetV2**. Pemilihan ini didasarkan pada karakteristik dataset yang memiliki variasi intra-kelas (satu jenis bunga dengan bentuk berbeda) dan variasi antar-kelas (bunga beda jenis namun terlihat mirip), sehingga pelatihan model dari awal (*from scratch*) seringkali tidak efisien dan membutuhkan sumber daya komputasi besar. **MobileNetV2** menjadi solusi karena memiliki keseimbangan yang baik antara akurasi tinggi dan efisiensi komputasi (ringan), serta memanfaatkan teknik *Transfer Learning*.

## 2. Tujuan Penelitian
Tujuan utama dari proyek akhir ini adalah:
1.  **Implementasi Deep Learning:** Menerapkan algoritma CNN untuk mengklasifikasikan 102 jenis bunga secara otomatis.
2.  **Penerapan Transfer Learning:** Mengimplementasikan teknik *Transfer Learning* menggunakan *pre-trained model* MobileNetV2 untuk meningkatkan akurasi dan mempercepat proses pelatihan dibandingkan melatih dari nol.
3.  **Evaluasi Kinerja:** Mengukur performa model menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*.
4.  **Analisis Model:** Menganalisis hasil pelatihan untuk mendeteksi indikasi *overfitting* atau *underfitting* serta kemampuan generalisasi model terhadap data baru.

---

## 3. Deskripsi Dataset
Kami menggunakan **Oxford 102 Flower Dataset**, sebuah dataset standar untuk *fine-grained classification*.

* **Sumber:** [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (diakses via `tensorflow_datasets`).
* **Jumlah Kelas:** 102 Kategori Spesies Bunga.
* **Format Data:** Gambar berwarna (RGB) dengan resolusi beragam.
* **Pembagian Data:** Dataset dibagi otomatis menjadi *Training Set*, *Validation Set*, dan *Test Set*.

### Tahap Pra-pemrosesan (Preprocessing)
Sebelum masuk ke model, data melalui tahapan berikut:
1.  **Resizing:** Mengubah seluruh gambar menjadi ukuran tetap **224 x 224 pixel** sesuai input MobileNetV2.
2.  **Normalisasi:** Mengonversi nilai pixel dari rentang 0-255 menjadi **0-1** agar komputasi model lebih efisien.
3.  **Batching:** Data dikelompokkan dalam *batch* (ukuran 32) untuk mempercepat pelatihan di GPU.

![SampleAcak](/readme/sampleacak.png "Sampel Gambar Acak")
---

## 4. Metodologi dan Arsitektur Model
Proyek ini menggunakan pendekatan **Transfer Learning** dengan detail arsitektur sebagai berikut:

### Arsitektur Jaringan
1.  **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
    * *Status:* Layer ini dibekukan (*Frozen*) agar bobot fitur dasar tidak berubah.
2.  **Custom Head (Top Layers):**
    * `GlobalAveragePooling2D`: Meratakan *feature map* menjadi vektor.
    * `Dropout(0.2)`: Menonaktifkan 20% neuron secara acak untuk mencegah *overfitting*.
    * `Dense(102, activation='softmax')`: Layer output dengan 102 neuron untuk prediksi probabilitas setiap kelas.

## 5. Hasil Eksperimen dan Analisis
Bagian ini memaparkan hasil evaluasi model serta analisis mendalam mengenai performa yang didapatkan.

### A. Evaluasi Akurasi dan Loss
Berikut adalah grafik perbandingan antara data *Training* dan *Validation* selama 10 Epoch:

![Grafik Training](https://via.placeholder.com/800x300?text=Screenshot+Grafik+Loss+dan+Accuracy)

**Analisis:**
1.  **Performa Umum:** Model berhasil mencapai akurasi validasi sekitar **80%**. Ini menunjukkan bahwa metode *Transfer Learning* dengan MobileNetV2 sangat efektif untuk dataset Oxford 102, mengingat melatih model dari awal (*scratch*) sangat sulit mencapai angka ini.
2.  **Indikasi Overfitting:** Terlihat garis akurasi Training (Biru) melesat ke angka ~99%, sedangkan Validasi (Oranye) tertahan di 80%. Adanya jarak (*gap*) ini menandakan *overfitting*, yaitu model mulai menghafal data latih. Namun, berkat penggunaan `Dropout(0.2)`, *overfitting* ini masih dalam batas wajar dan model tetap bisa memprediksi data baru dengan baik.

---

### B. Evaluasi Detail (Classification Report)
Tabel berikut menunjukkan performa model pada data uji (*Test Set*):

| Metric | Score | Keterangan |
| :--- | :--- | :--- |
| **Accuracy** | **80%** | Tingkat kebenaran keseluruhan |
| **Precision** | 81% | Ketepatan tebakan positif |
| **Recall** | 80% | Kemampuan menemukan semua sampel positif |
| **F1-Score** | 80% | Rata-rata harmonis Precision & Recall |

**Analisis:**
Nilai Precision dan Recall yang seimbang menunjukkan bahwa model tidak bias ke satu kelas tertentu saja. Model cukup "adil" dalam mengenali ke-102 jenis bunga.

---

### C. Analisis Kesalahan (Confusion Matrix)
Untuk melihat di mana letak kesalahan model, kita menggunakan Confusion Matrix:

<div align="center">
  <img src="https://via.placeholder.com/600x600?text=Screenshot+Confusion+Matrix" alt="Confusion Matrix" width="600"/>
</div>

**Analisis:**
* **Diagonal Utama:** Garis diagonal yang berwarna gelap menunjukkan bahwa mayoritas prediksi model sudah benar (Label Asli = Prediksi).
* **Kesalahan Prediksi:** Bercak-bercak samar di luar diagonal menunjukkan kesalahan. Hal ini wajar terjadi pada jenis bunga yang memiliki kemiripan visual sangat tinggi (misalnya antara *'pink primrose'* dan *'primula'*), yang bahkan sulit dibedakan oleh mata manusia.
