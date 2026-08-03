# Coffee Quality Detection Backend

Backend FastAPI untuk deteksi kualitas biji kopi menggunakan Ultralytics YOLO. API ini dipakai oleh aplikasi Flutter dan tetap mempertahankan field response lama seperti `success`, `data`, `class_name`, `coffee_type`, `grade`, `confidence`, `status`, `recommendation`, `characteristics`, dan `bounding_boxes`.

## Fitur

- Prediksi kualitas biji kopi dengan YOLOv11.
- Confidence threshold default 0.50 untuk mengurangi false detection pada background atau objek non-kopi.
- Validasi upload gambar: ekstensi, MIME type, ukuran maksimal, dan verifikasi Pillow.
- Auth register, login, Google login, JWT Bearer token.
- Role `admin` dan `user`.
- Admin-only upload/update model `best.pt`.
- Admin dapat melihat daftar user, status aktif, status online, dan total prediksi.
- Riwayat prediksi tersimpan di SQLite.
- Password bcrypt dengan fallback SHA-256 lama yang di-upgrade otomatis saat login.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Contoh `.env`

```env
CONFIDENCE_THRESHOLD=0.5
MAX_IMAGE_SIZE_MB=5
MAX_MODEL_SIZE_MB=200

ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=ganti-password-admin

CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8080

JWT_SECRET_KEY=ganti-secret-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

MODEL_PATH=models/best.pt
UPLOAD_DIR=uploads
ONLINE_USER_WINDOW_MINUTES=10

GOOGLE_CLIENT_ID=
```

Upload model hanya menerima `Authorization: Bearer <JWT>` dari akun admin yang
masih aktif. Header `X-Admin-Token` tidak didukung.

File `.pt` hanya boleh berasal dari sumber yang dipercaya. Validasi struktur
model tidak menjamin checkpoint dari sumber tidak terpercaya aman untuk
dideserialisasi. Untuk aplikasi skripsi ini, upload dibatasi hanya kepada akun
admin aktif. Migrasi ke format seperti ONNX dapat dipertimbangkan sebagai
pengembangan keamanan lanjutan.

## Membuat Admin Awal

Isi `ADMIN_EMAIL` dan `ADMIN_PASSWORD` di `.env`, lalu jalankan backend. Saat startup, jika belum ada user role admin, backend akan membuat admin dari env tersebut. Endpoint register biasa selalu membuat role `user`, bukan admin.

## Auth

Register user biasa:

```bash
curl -X POST http://localhost:8000/auth/register ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"User\",\"email\":\"user@example.com\",\"password\":\"password123\"}"
```

Login admin:

```bash
curl -X POST http://localhost:8000/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"email\":\"admin@example.com\",\"password\":\"ganti-password-admin\"}"
```

Response login menyertakan `access_token` dan `data.role`.

## Upload Model Sebagai Admin

```bash
curl -X POST http://localhost:8000/models/upload ^
  -H "Authorization: Bearer <ADMIN_TOKEN>" ^
  -F "file=@models/best.pt"
```

Jika user biasa mencoba upload model:

```json
{
  "success": false,
  "status": "error",
  "message": "Akses ditolak. Hanya admin yang dapat mengunggah model.",
  "detections": [],
  "total_detected": 0
}
```

File model diupload ke temporary path, divalidasi dengan `YOLO(temp_path)`, lalu baru mengganti `models/best.pt`. Jika validasi gagal, model lama tidak ditimpa.

## Daftar User Aktif

```bash
curl http://localhost:8000/admin/users ^
  -H "Authorization: Bearer <ADMIN_TOKEN>"
```

Contoh response:

```json
{
  "status": "success",
  "users": [
    {
      "id": "uuid",
      "name": "Admin",
      "email": "admin@example.com",
      "role": "admin",
      "is_active": true,
      "is_online": true,
      "last_login_at": "2026-07-01T10:00:00+00:00",
      "last_seen_at": "2026-07-01T10:05:00+00:00",
      "total_predictions": 12
    }
  ]
}
```

User dianggap online jika `last_seen_at` masih dalam `ONLINE_USER_WINDOW_MINUTES`, default 10 menit.

## Predict

Endpoint `/predict` tetap dapat dipakai tanpa token agar frontend lama tidak rusak. Jika `Authorization: Bearer <token>` dikirim, backend menyimpan `user_id` pada riwayat prediksi dan memperbarui `last_seen_at`.

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Authorization: Bearer <TOKEN_OPSIONAL>" ^
  -F "file=@sample.jpg"
```

Validasi gambar:

- Ekstensi hanya `.jpg`, `.jpeg`, `.png`.
- MIME type hanya `image/jpeg` atau `image/png`.
- File harus bisa dibuka sebagai gambar dengan Pillow.
- Ukuran maksimal default 5 MB.

## Response Detected

```json
{
  "success": true,
  "status": "detected",
  "message": "Biji kopi berhasil terdeteksi.",
  "detections": [
    {
      "class_name": "Arabica Grade A",
      "coffee_type": "Arabica",
      "grade": "Grade A",
      "confidence": 0.91,
      "bbox": {
        "x": 0.12,
        "y": 0.2,
        "width": 0.4,
        "height": 0.3,
        "confidence": 0.91
      },
      "recommendation": "Layak jual kualitas tinggi.",
      "characteristics": {
        "bentuk_keutuhan": "Biji utuh dan bentuk relatif seragam.",
        "ukuran": "Ukuran biji relatif seragam.",
        "permukaan": "Permukaan biji halus dan baik.",
        "warna": "Warna biji merata dan tidak terdapat cacat mencolok."
      },
      "detected_at": "2026-07-01T10:05:00+00:00"
    }
  ],
  "total_detected": 1,
  "confidence_threshold": 0.5,
  "data": {
    "class_name": "Arabica Grade A",
    "coffee_type": "Arabica",
    "grade": "Grade A",
    "confidence": 0.91,
    "status": "Kualitas Tinggi",
    "bounding_boxes": []
  }
}
```

## Response Not Detected

```json
{
  "success": true,
  "status": "not_detected",
  "message": "Tidak ada biji kopi terdeteksi. Silakan ambil gambar ulang dengan pencahayaan yang cukup dan objek biji kopi terlihat jelas.",
  "detections": [],
  "total_detected": 0,
  "confidence_threshold": 0.5,
  "data": {
    "class_name": "Tidak Terdeteksi",
    "coffee_type": "-",
    "grade": "-",
    "confidence": 0,
    "status": "Tidak Terdeteksi",
    "bounding_boxes": []
  }
}
```

## Response Error File Invalid

```json
{
  "success": false,
  "status": "error",
  "message": "File gambar tidak valid atau tidak dapat diproses.",
  "detections": [],
  "total_detected": 0
}
```

## Catatan Confidence Threshold

`CONFIDENCE_THRESHOLD=0.5` dikirim ke `model.predict(..., conf=settings.confidence_threshold)` dan hasil box tetap difilter ulang di backend. Jika tidak ada detection dengan confidence minimal 0.50, backend mengembalikan `status: not_detected` dan tidak membuat klasifikasi palsu.

## Role

- `user`: role default dari register biasa.
- `admin`: hanya dibuat lewat env admin awal atau data database yang sudah diset admin.
- `GET /admin/users` dan `POST /models/upload` wajib JWT admin.
