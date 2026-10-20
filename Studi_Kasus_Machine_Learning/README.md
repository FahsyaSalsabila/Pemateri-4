# Studi Kasus Machine Learning: Deteksi Pesan Spam

Proyek sederhana ini menerapkan **supervised learning** untuk mengklasifikasikan pesan teks menjadi `spam` atau `bukan_spam`.

## Tujuan
Mempraktikkan alur kerja machine learning yang dibahas dalam materi:
1. Menyiapkan dan membersihkan data.
2. Memisahkan data menjadi training set (80%) dan testing set (20%).
3. Mengubah teks menjadi fitur angka menggunakan `CountVectorizer`.
4. Melatih model klasifikasi menggunakan `MultinomialNB` (Naive Bayes).
5. Mengevaluasi model dengan akurasi, classification report, dan confusion matrix.
6. Melakukan inference pada pesan baru.

## Struktur folder

```text
Studi_Kasus_Machine_Learning/
├── main.py
├── requirements.txt
└── README.md
```

## Persiapan di VS Code

Pastikan Python sudah terpasang. Buka folder ini di VS Code, lalu buka Terminal > New Terminal.

### 1. Buat virtual environment (disarankan)

Windows:
```powershell
py -m venv .venv
.venv\Scripts\activate
```

Jika perintah `py` tidak tersedia, coba `python -m venv .venv`.

### 2. Instal library

```powershell
python -m pip install -r requirements.txt
```

### 3. Jalankan program

```powershell
python main.py
```

Program akan menampilkan jumlah data pelatihan dan pengujian, akurasi, laporan evaluasi, confusion matrix, serta prediksi untuk beberapa pesan baru.

## Penjelasan singkat kode

- `bersihkan_teks()`: mengubah huruf menjadi huruf kecil dan menghapus tanda baca sederhana.
- `train_test_split()`: membagi data agar evaluasi dilakukan pada data yang tidak digunakan untuk melatih model.
- `CountVectorizer`: mengubah kata menjadi representasi angka.
- `MultinomialNB`: algoritma klasifikasi yang mempelajari hubungan kata dan label.
- `model.fit()`: proses pelatihan.
- `model.predict()`: membuat prediksi untuk pesan baru.
- `accuracy_score()`: menghitung proporsi prediksi yang benar pada data pengujian.

## Catatan penting

Dataset contoh sengaja dibuat kecil supaya mudah dipahami untuk tugas pengantar. Nilai akurasi dapat berubah jika dataset atau pembagian datanya diubah. Akurasi tinggi pada dataset kecil **tidak** berarti model sudah akurat untuk semua pesan nyata. Untuk proyek yang lebih serius, gunakan dataset yang lebih besar dan beragam serta evaluasi tambahan.

## Cara upload ke GitHub

1. Buat akun atau masuk ke [GitHub](https://github.com/).
2. Klik **New repository**.
3. Isi nama repository, misalnya `studi-kasus-machine-learning`.
4. Pilih Public atau Private, lalu klik **Create repository**.
5. Di VS Code, buka Terminal pada folder proyek ini. Pastikan Git sudah terpasang.
6. Jalankan perintah berikut (ganti URL dengan URL repository milikmu):

```bash
git init
git add main.py requirements.txt README.md
git commit -m "Tambahkan studi kasus klasifikasi spam"
git branch -M main
git remote add origin https://github.com/USERNAME/studi-kasus-machine-learning.git
git push -u origin main
```

Ganti `USERNAME` dengan username GitHub kamu. Jika Git meminta identitas, atur nama dan email Git dengan perintah `git config --global user.name "Nama Kamu"` dan `git config --global user.email "email-kamu@example.com"`.

**Jangan pernah mengunggah password, token akses, atau informasi rahasia ke repository.**
