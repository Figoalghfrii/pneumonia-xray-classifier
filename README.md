# Pneumonia X-Ray Classifier

Web app sederhana berbasis Deep Learning untuk klasifikasi citra X-Ray dada menjadi dua kelas: `NORMAL` dan `PNEUMONIA`.

Project ini menggunakan Transfer Learning dengan MobileNetV2 karena relatif ringan, efektif, dan cocok untuk deployment web app. Aplikasi dibuat menggunakan Streamlit.

## Dataset

Dataset yang digunakan:

[Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Dataset berisi citra X-Ray dada dalam dua kelas:

- `NORMAL`
- `PNEUMONIA`

## Struktur Project

```text
pneumonia-xray-classifier/
|-- app.py
|-- train_model.py
|-- pneumonia_model.keras
|-- requirements.txt
|-- README.md
|-- utils/
|   `-- preprocessing.py
`-- dataset/
    `-- chest_xray/
        |-- train/
        |   |-- NORMAL/
        |   `-- PNEUMONIA/
        |-- test/
        |   |-- NORMAL/
        |   `-- PNEUMONIA/
        `-- val/
            |-- NORMAL/
            `-- PNEUMONIA/
```

Catatan: file `pneumonia_model.keras` akan dibuat setelah proses training selesai.

## Cara Download Dataset dari Kaggle

1. Buka halaman dataset Kaggle:
   <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>
2. Klik tombol download.
3. Ekstrak file dataset.
4. Salin folder `chest_xray` ke dalam folder `dataset/` project ini.

Struktur akhir dataset harus seperti ini:

```text
dataset/
`-- chest_xray/
    |-- train/
    |   |-- NORMAL/
    |   `-- PNEUMONIA/
    |-- test/
    |   |-- NORMAL/
    |   `-- PNEUMONIA/
    `-- val/
        |-- NORMAL/
        `-- PNEUMONIA/
```

## Instalasi

Buat virtual environment, lalu install dependency:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Untuk Linux atau macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Model

Jalankan:

```bash
python train_model.py
```

Script training akan:

- Membaca dataset dari `dataset/chest_xray`
- Resize gambar ke ukuran `224x224`
- Melakukan normalisasi input MobileNetV2
- Menggunakan data augmentation untuk training
- Membangun model Transfer Learning MobileNetV2
- Menggunakan binary classification dengan output sigmoid
- Menggunakan loss `binary_crossentropy`
- Menggunakan optimizer Adam
- Menggunakan callback `EarlyStopping`, `ModelCheckpoint`, dan `ReduceLROnPlateau`
- Mengevaluasi model dengan accuracy, precision, recall, F1-score, dan confusion matrix
- Menyimpan model ke `pneumonia_model.keras`

Output tambahan:

- `confusion_matrix.png`

## Menjalankan Web App

Setelah model selesai dibuat, jalankan:

```bash
streamlit run app.py
```

Upload gambar X-Ray dada dalam format JPG, JPEG, atau PNG, lalu tekan tombol prediksi.

## Cara Deployment ke Streamlit Community Cloud

1. Push project ke GitHub.
2. Login ke [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Pilih repository project.
4. Set main file ke `app.py`.
5. Deploy aplikasi.

Pastikan file `pneumonia_model.keras` sudah ada di repository atau tersedia saat deployment. Jika ukuran model terlalu besar untuk GitHub, gunakan Git LFS atau hosting model eksternal, lalu sesuaikan kode loading model.

## Contoh Penggunaan

1. Jalankan aplikasi dengan `streamlit run app.py`.
2. Upload citra X-Ray dada.
3. Lihat preview gambar.
4. Klik tombol `Prediksi`.
5. Aplikasi akan menampilkan hasil `NORMAL` atau `PNEUMONIA` beserta confidence score.

## Hasil Evaluasi

Model dievaluasi menggunakan test set dengan threshold prediksi 0.7.

| Metric               | Score |
--------------------------------
| Accuracy              | 82% |

| Precision Pneumonia   | 79% |

| Recall Pneumonia      | 98% |

| F1-score Pneumonia    | 87% |

## Insight

Model memiliki recall yang sangat tinggi pada kelas `PNEUMONIA`, yaitu 98%. Artinya, sebagian besar citra X-Ray dengan pneumonia berhasil terdeteksi oleh model.

Namun, recall untuk kelas `NORMAL` masih 56%, sehingga masih terdapat citra normal yang diklasifikasikan sebagai pneumonia. Dalam konteks deteksi awal pneumonia, kesalahan seperti ini relatif lebih dapat diterima dibandingkan false negative, tetapi model tetap belum layak digunakan sebagai alat diagnosis medis.

Penyesuaian threshold dari 0.5 ke 0.7 membantu meningkatkan keseimbangan performa model, terutama dengan mengurangi jumlah prediksi pneumonia yang salah pada citra normal.

## Limitasi

- Dataset berasal dari citra X-Ray pasien anak-anak, sehingga belum tentu cocok untuk pasien dewasa.
- Dataset memiliki distribusi kelas yang tidak seimbang.
- Model hanya membedakan dua kelas: `NORMAL` dan `PNEUMONIA`.
- Model tidak membedakan pneumonia bakteri, virus, atau kondisi paru lainnya.
- Hasil prediksi tidak boleh digunakan sebagai pengganti diagnosis dokter.

## Disclaimer Medis

Model ini hanya untuk pembelajaran dan demonstrasi. Model tidak boleh digunakan sebagai pengganti diagnosis dokter. Dataset memiliki keterbatasan, sehingga hasil prediksi tidak selalu akurat. Untuk keputusan medis, selalu konsultasikan dengan tenaga medis profesional.
