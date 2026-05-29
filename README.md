# Coffee Quality YOLO API

Backend FastAPI sementara untuk aplikasi Flutter skripsi:

**Klasifikasi Kualitas Biji Kopi Berdasarkan Citra Digital Dengan Menggunakan Metode YOLOv11**

Backend ini masih skeleton/dummy. Model YOLOv11 asli belum diintegrasikan karena dataset masih dalam proses labeling/training. Setelah training selesai, file model dapat diletakkan di:

```text
models/best.pt
```

## Fitur Saat Ini

- `GET /health` untuk cek status API.
- `POST /predict` menerima upload gambar `jpg`, `jpeg`, atau `png`.
- File upload disimpan ke folder `uploads/`.
- Response prediction masih dummy random dari 6 kelas:
  - Arabica Grade A
  - Arabica Grade B
  - Arabica Grade C
  - Robusta Grade A
  - Robusta Grade B
  - Robusta Grade C

Belum ada database, authentication, atau model YOLO asli.

## Instalasi

```powershell
python -m venv venv
```

```powershell
venv\Scripts\activate
```

```powershell
pip install -r requirements.txt
```

## Menjalankan Server

Jalankan dari folder `coffee_backend`:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Buka dokumentasi API:

```text
http://127.0.0.1:8000/docs
```

Cek health endpoint:

```text
http://127.0.0.1:8000/health
```

## Contoh Response `/predict`

```json
{
  "success": true,
  "message": "Prediction completed",
  "data": {
    "image_name": "uploaded-file.png",
    "class_name": "Arabica Grade A",
    "coffee_type": "Arabica",
    "grade": "Grade A",
    "confidence": 0.925,
    "confidence_percent": 92.5,
    "status": "Kualitas Tinggi",
    "description": "Biji kopi terdeteksi sebagai Arabica Grade A dengan kualitas tinggi.",
    "recommendation": "Layak jual kualitas tinggi.",
    "characteristics": {
      "bentuk_keutuhan": "Biji utuh dan bentuk relatif seragam.",
      "ukuran": "Ukuran biji relatif seragam.",
      "permukaan": "Permukaan biji halus dan baik.",
      "warna": "Warna biji merata dan tidak terdapat cacat mencolok."
    },
    "bounding_boxes": [
      {
        "x": 0.35,
        "y": 0.22,
        "width": 0.3,
        "height": 0.45,
        "confidence": 0.925,
        "label": "Arabica Grade A",
        "class_name": "Arabica Grade A",
        "coffee_type": "Arabica",
        "grade": "Grade A"
      },
      {
        "x": 0.62,
        "y": 0.3,
        "width": 0.18,
        "height": 0.24,
        "confidence": 0.881,
        "label": "Arabica Grade A",
        "class_name": "Arabica Grade A",
        "coffee_type": "Arabica",
        "grade": "Grade A"
      }
    ]
  }
}
```

`class_name`, `coffee_type`, `grade`, dan `confidence` di level `data` tetap
mengikuti deteksi dengan confidence tertinggi. Semua objek hasil deteksi YOLO
dikirim di `bounding_boxes`.

## Integrasi Model YOLOv11 Nanti

Untuk integrasi model asli, backend kemungkinan akan membutuhkan package tambahan:

```text
ultralytics
opencv-python
pillow
```

TODO utama ada di `app/services/yolo_service.py`: load `models/best.pt` menggunakan `ultralytics.YOLO`, jalankan inference pada gambar upload, lalu ubah hasil inference ke format JSON yang sama agar Flutter tidak perlu banyak berubah.
