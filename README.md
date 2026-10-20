# Studi Kasus Machine Learning: Klasifikasi Pesan Spam

## 1. Latar Belakang

Pesan spam merupakan pesan yang tidak diinginkan, seperti promosi berlebihan, penawaran hadiah, atau pesan mencurigakan. Banyaknya pesan tersebut dapat mengganggu pengguna. Oleh karena itu, diperlukan sistem yang dapat membantu membedakan pesan spam dan pesan biasa secara otomatis.

Machine Learning (ML) merupakan bagian dari Artificial Intelligence (AI) yang memungkinkan komputer mempelajari pola dari data untuk menghasilkan prediksi. Pada studi kasus ini, Machine Learning diterapkan untuk mengklasifikasikan pesan ke dalam dua kategori, yaitu **spam** dan **bukan spam**.

## 2. Rumusan Masalah

1. Bagaimana menerapkan Machine Learning untuk mengklasifikasikan pesan spam dan bukan spam?
2. Bagaimana melatih model menggunakan data pesan yang telah diberi label?
3. Bagaimana mengevaluasi kemampuan model dalam melakukan klasifikasi?

## 3. Tujuan

1. Memahami penerapan konsep *supervised learning* dalam permasalahan nyata.
2. Membuat model klasifikasi teks menggunakan Python.
3. Melatih model menggunakan dataset pesan.
4. Menguji kinerja model menggunakan data pengujian.
5. Menghasilkan prediksi untuk pesan baru.

## 4. Metode yang Digunakan

Studi kasus ini menggunakan pendekatan *supervised learning*, yaitu metode pembelajaran mesin yang memanfaatkan data berlabel.

| Komponen | Keterangan |
|---|---|
| Bidang | Machine Learning |
| Jenis pembelajaran | Supervised Learning |
| Tugas | Classification |
| Algoritma | Multinomial Naive Bayes |
| Bahasa pemrograman | Python |
| Library | Scikit-learn |
| Input | Teks pesan |
| Output | Spam atau bukan spam |

Algoritma Multinomial Naive Bayes dipilih karena relatif sederhana dan dapat digunakan untuk klasifikasi teks berdasarkan pola kemunculan kata.

## 5. Dataset

Dataset yang digunakan berisi contoh pesan yang telah diberi label. Dataset ini dibuat untuk keperluan pembelajaran dan demonstrasi program.

Contoh data:

| No. | Pesan | Label |
|---|---|---|
| 1 | Selamat, Anda memenangkan hadiah uang tunai! | Spam |
| 2 | Klaim hadiah gratis sekarang juga | Spam |
| 3 | Pinjaman cepat cair, daftar sekarang | Spam |
| 4 | Jangan lupa membawa buku untuk kuliah besok | Bukan spam |
| 5 | Tolong kirim catatan materi Machine Learning | Bukan spam |
| 6 | Aku sudah sampai di depan rumahmu | Bukan spam |

Program menggunakan 24 contoh pesan yang terdiri dari kategori spam dan bukan spam. Dataset berukuran kecil ini digunakan untuk mempermudah pemahaman terhadap proses pelatihan dan pengujian model.

## 6. Implementasi Program

Program dibuat menggunakan Python dengan bantuan library Scikit-learn. Tahapan implementasinya adalah sebagai berikut.

### A. Persiapan dan Pembersihan Data

Data pesan disiapkan beserta labelnya. Teks kemudian dibersihkan agar formatnya lebih konsisten, misalnya dengan mengubah huruf menjadi huruf kecil dan menghapus tanda baca yang tidak diperlukan.

### B. Pembagian Dataset

Dataset dibagi menjadi dua bagian:

- **Training set (80%)**: digunakan untuk melatih model.
- **Testing set (20%)**: digunakan untuk menguji model menggunakan data yang tidak dipakai selama pelatihan.

Pembagian ini membantu menilai kemampuan model dalam mengklasifikasikan data yang belum dipelajari.

### C. Pengubahan Teks Menjadi Fitur

Teks pesan diubah menjadi representasi numerik menggunakan `CountVectorizer`. Proses ini memungkinkan algoritma memproses kata-kata sebagai fitur dalam model Machine Learning.

### D. Pelatihan Model

Model dilatih menggunakan algoritma `MultinomialNB`. Pada tahap ini, model mempelajari hubungan antara fitur kata dan label spam atau bukan spam dari data pelatihan.

### E. Pengujian Model

Setelah pelatihan selesai, model digunakan untuk memprediksi label pada data pengujian. Hasil prediksi kemudian dibandingkan dengan label sebenarnya untuk mengevaluasi kinerja model.

### F. Prediksi Data Baru

Model yang telah dilatih digunakan untuk memprediksi kategori pesan baru yang belum terdapat dalam dataset pelatihan.

Kode program lengkap dapat dilihat pada file [`main.py`](main.py).

## 7. Alur Kerja Machine Learning

Alur kerja dalam studi kasus ini meliputi:

1. Mengumpulkan dan menyiapkan dataset pesan.
2. Membersihkan data teks.
3. Membagi dataset menjadi training set dan testing set.
4. Mengubah teks menjadi fitur numerik.
5. Melatih model menggunakan Multinomial Naive Bayes.
6. Mengevaluasi model menggunakan data pengujian.
7. Melakukan *inference* atau prediksi pada pesan baru.

Alur tersebut menunjukkan penerapan workflow Machine Learning mulai dari pengolahan data hingga menghasilkan prediksi.

## 8. Pengujian dan Evaluasi

Evaluasi dilakukan untuk mengetahui kemampuan model dalam mengklasifikasikan pesan. Program menggunakan beberapa metrik evaluasi berikut.

- **Accuracy**: mengukur persentase prediksi yang benar dari seluruh data pengujian.
- **Precision**: mengukur proporsi prediksi spam yang benar-benar merupakan spam.
- **Recall**: mengukur proporsi pesan spam yang berhasil dikenali oleh model.
- **Confusion Matrix**: menunjukkan jumlah prediksi benar dan salah untuk setiap kategori.

Hasil evaluasi ditampilkan ketika program dijalankan. Nilai akurasi dan metrik lainnya tidak ditentukan terlebih dahulu karena bergantung pada hasil prediksi model terhadap data pengujian.

## 9. Hasil yang Diharapkan

Program diharapkan dapat mempelajari pola kata dari contoh pesan dan menghasilkan prediksi berupa kategori spam atau bukan spam untuk pesan baru.

Sebagai contoh, pesan yang mengandung penawaran hadiah dapat diprediksi sebagai spam, sedangkan pesan percakapan sehari-hari dapat diprediksi sebagai bukan spam. Namun, hasil tersebut merupakan prediksi model dan tidak selalu benar.

## 10. Kesimpulan

Studi kasus ini menunjukkan penerapan Machine Learning untuk membantu mengklasifikasikan pesan spam dan bukan spam. Dengan pendekatan *supervised learning* dan algoritma Multinomial Naive Bayes, model mempelajari pola dari data berlabel untuk menghasilkan prediksi pada pesan baru.

Proses pengembangan mencakup persiapan data, pembersihan teks, pembagian dataset, pelatihan model, evaluasi, dan *inference*. Seluruh tahapan tersebut sesuai dengan workflow Machine Learning yang dipelajari dalam materi *Introduction to Machine Learning*.

Meskipun demikian, dataset yang digunakan masih terbatas sehingga model ini ditujukan untuk pembelajaran. Pengembangan lebih lanjut memerlukan dataset yang lebih besar dan beragam agar kinerja model dapat dievaluasi dengan lebih baik.

