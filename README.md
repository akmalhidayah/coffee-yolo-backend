
---

# README Backend — `coffee-yolo-backend`

```markdown
# Coffee Quality Detection - Backend

Coffee Quality Detection Backend adalah layanan REST API yang digunakan untuk memproses deteksi dan klasifikasi kualitas biji kopi berdasarkan citra digital menggunakan model YOLOv11. Backend ini dikembangkan menggunakan FastAPI dan berfungsi sebagai penghubung antara aplikasi Flutter dengan model deteksi YOLO.

Backend menerima input berupa gambar biji kopi dari frontend, melakukan proses prediksi menggunakan model YOLOv11, kemudian mengembalikan hasil klasifikasi dalam bentuk JSON.

## Fitur Utama

Backend ini memiliki beberapa fitur utama, yaitu:

- REST API untuk prediksi kualitas biji kopi.
- Integrasi model YOLOv11 untuk deteksi objek dan klasifikasi grade.
- Endpoint health check untuk mengecek status server.
- Autentikasi pengguna melalui register dan login.
- Login menggunakan Google.
- Manajemen data pengguna.
- Upload model YOLO berformat `.pt`.
- Penyimpanan file gambar hasil upload.
- Response prediksi lengkap berupa jenis kopi, grade, confidence score, status kualitas, karakteristik fisik, rekomendasi, dan bounding box.

## Teknologi yang Digunakan

Backend aplikasi ini dikembangkan menggunakan teknologi berikut:

- Python
- FastAPI
- Uvicorn
- Ultralytics YOLO
- SQLite
- Pydantic
- Python Multipart
- JWT Authentication

## Struktur Folder

Struktur utama folder backend adalah sebagai berikut:

```text
coffee-yolo-backend/
├── app/
│   ├── core/          # Konfigurasi aplikasi
│   ├── routes/        # Endpoint API
│   ├── services/      # Service prediksi, user, dan model
│   └── main.py        # Entry point FastAPI
├── models/            # Folder penyimpanan model YOLO
├── uploads/           # Folder penyimpanan gambar yang diunggah
├── requirements.txt   # Dependency Python
└── README.md
