"""
Studi Kasus Machine Learning:
Klasifikasi pesan SMS menjadi SPAM atau BUKAN SPAM.

Materi yang digunakan:
- Supervised learning
- Classification
- Data cleaning sederhana
- Training set dan testing set
- Evaluasi akurasi
- Inference pada data baru
"""

import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Dataset contoh buatan untuk demonstrasi pembelajaran.
# Label: "spam" atau "bukan_spam"
data = [
    ("Selamat! Anda memenangkan hadiah uang tunai, klik tautan ini", "spam"),
    ("Klaim hadiah gratis sekarang juga", "spam"),
    ("Akun Anda terpilih mendapatkan bonus besar", "spam"),
    ("Pinjaman cepat cair tanpa jaminan, daftar sekarang", "spam"),
    ("Promo terbatas! Beli sekarang dan dapatkan diskon besar", "spam"),
    ("Anda mendapat voucher gratis, tekan tautan untuk klaim", "spam"),
    ("Menangkan hadiah jutaan rupiah hari ini", "spam"),
    ("Bonus spesial menunggu Anda, klaim sekarang", "spam"),
    ("Transfer biaya administrasi untuk menerima hadiah", "spam"),
    ("Peluang investasi pasti untung, daftar sekarang", "spam"),
    ("Besok kita jadi belajar kelompok di perpustakaan?", "bukan_spam"),
    ("Jangan lupa bawa buku untuk kuliah besok", "bukan_spam"),
    ("Aku sudah sampai di depan rumahmu", "bukan_spam"),
    ("Tolong kirim catatan materi machine learning", "bukan_spam"),
    ("Rapat kelompok dimulai pukul tiga sore", "bukan_spam"),
    ("Terima kasih sudah membantu tugas tadi", "bukan_spam"),
    ("Nanti malam kita mengerjakan laporan bersama", "bukan_spam"),
    ("Dosen mengumumkan jadwal presentasi hari Jumat", "bukan_spam"),
    ("Aku sedang dalam perjalanan menuju kampus", "bukan_spam"),
    ("Jangan lupa makan siang sebelum kelas", "bukan_spam"),
    ("File tugas sudah aku kirim melalui email", "bukan_spam"),
    ("Apakah kamu bisa hadir pada diskusi besok?", "bukan_spam"),
    ("Kelas hari ini selesai lebih awal", "bukan_spam"),
    ("Sampai jumpa di laboratorium komputer", "bukan_spam"),
]


def bersihkan_teks(teks):
    """Mengubah teks menjadi huruf kecil dan menghapus tanda baca berlebih."""
    teks = teks.lower()
    teks = re.sub(r"[^a-zA-Z0-9\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def main():
    pesan = [bersihkan_teks(item[0]) for item in data]
    label = [item[1] for item in data]

    # 80% data untuk pelatihan dan 20% untuk pengujian.
    X_train, X_test, y_train, y_test = train_test_split(
        pesan,
        label,
        test_size=0.2,
        random_state=42,
        stratify=label,
    )

    # Pipeline mengubah teks menjadi angka, lalu melatih model klasifikasi.
    model = Pipeline([
        ("vektorisasi", CountVectorizer()),
        ("klasifikasi", MultinomialNB()),
    ])

    model.fit(X_train, y_train)

    # Evaluasi menggunakan data yang tidak dipakai saat pelatihan.
    prediksi = model.predict(X_test)
    akurasi = accuracy_score(y_test, prediksi)

    print("=" * 55)
    print("STUDI KASUS: DETEKSI PESAN SPAM")
    print("=" * 55)
    print(f"Jumlah seluruh data : {len(data)}")
    print(f"Data pelatihan      : {len(X_train)}")
    print(f"Data pengujian      : {len(X_test)}")
    print(f"Akurasi pengujian   : {akurasi:.2%}")
    print("\nLaporan klasifikasi:")
    print(classification_report(y_test, prediksi, zero_division=0))
    print("Confusion matrix (urutan label: bukan_spam, spam):")
    print(confusion_matrix(y_test, prediksi, labels=["bukan_spam", "spam"]))

    # Inference: mencoba model pada pesan baru.
    pesan_baru = [
        "Selamat Anda mendapatkan hadiah gratis, klaim sekarang",
        "Jangan lupa mengirim laporan tugas malam ini",
        "Promo diskon besar berlaku sampai besok",
    ]

    print("\nHasil prediksi untuk pesan baru:")
    for teks in pesan_baru:
        hasil = model.predict([bersihkan_teks(teks)])[0]
        nama_label = "SPAM" if hasil == "spam" else "BUKAN SPAM"
        print(f"- Pesan: {teks}")
        print(f"  Prediksi: {nama_label}\n")

    print("Catatan: dataset ini kecil dan dibuat untuk latihan.")
    print("Model belum layak digunakan sebagai penyaring spam sungguhan.")


if __name__ == "__main__":
    main()
