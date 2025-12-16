# Biji Hitam Coffee Classification API

API untuk klasifikasi jenis biji kopi Indonesia menggunakan deep learning model ConvNeXt. Mendukung 54 varian kopi (Arabika, Liberika, dan Robusta).

## Instalasi

```bash
git clone <repository-url>
cd biji-hitam-backend

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Pastikan file `best_model.pth` ada di direktori root.

## Menjalankan API

```bash
# Menggunakan fastapi dev (recommended)
fastapi dev main.py

# Atau menggunakan uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Buka dokumentasi API di: `http://localhost:8000/docs`

## API Endpoints

### GET /

Health check - mengembalikan status API dan informasi model.

### POST /predict

Upload gambar untuk prediksi jenis kopi.

**Request:**

- `file`: Gambar (JPG, PNG, WebP, dll) - maksimal 2MB

**Response:**

```json
{
  "predictions": [
    {
      "class_name": "Arabika Aceh Gayo",
      "confidence": 95.32,
      "index": 0
    },
    ...
  ]
}
```

## Contoh Penggunaan

### cURL

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@kopi.jpg"
```

### Python

```python
import requests

with open("kopi.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/predict", files={"file": f})
    print(response.json())
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData,
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

## Jenis Kopi

- Arabika: 37 varian
- Liberika: 3 varian
- Robusta: 14 varian

## Konfigurasi

Edit di `main.py`:

- `MODEL_PATH`: Path ke model file
- `MAX_FILE_SIZE`: Ukuran file maksimal (2 MB)
- `NUM_CLASSES`: Jumlah kelas (54)  

## Troubleshooting

- **Model not loaded**: Pastikan `best_model.pth` ada di root
- **File size exceeds limit**: Kompres gambar atau ubah `MAX_FILE_SIZE`
- **Prediksi tidak akurat**: Gunakan gambar close-up dengan pencahayaan baik

## Requirements

```
fastapi[standard]
torch
torchvision
timm
python-multipart
pillow
```

---

Made with Python and PyTorch
