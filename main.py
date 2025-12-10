import torch
import timm
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Konfigurasi
MODEL_PATH = "best_model.pth" 
MODEL_NAME = 'convnext_tiny.in12k_ft_in1k'
NUM_CLASSES = 54
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

CLASS_NAMES = ['Arabika Aceh Gayo', 'Arabika Argo Puro', 'Arabika Arjuno Budug Asu', 'Arabika Bali Batu Karu', 'Arabika Bali Kintamani', 'Arabika Bali Ulian', 'Arabika Dolok Sanggul', 'Arabika Enrekang', 'Arabika Flores Bajawa', 'Arabika Flores Manggarai', 'Arabika Golosera Wae Rebo', 'Arabika Ijen', 'Arabika Java Ciwidey', 'Arabika Java Garut', 'Arabika Java Puntang', 'Arabika Java Temanggung', 'Arabika Kerinci Kayu Aro', 'Arabika Latimojong', 'Arabika Lintong', 'Arabika Lintong Onan Ganjang', 'Arabika Malabar Mountain', 'Arabika Mandailing', 'Arabika Mangalayang', 'Arabika Papua Lembah Kamu', 'Arabika Papua Moanemani', 'Arabika Papua Yahukimo', 'Arabika Semendo', 'Arabika Simalungun', 'Arabika Sindoro', 'Arabika Sipirok', 'Arabika Solok Radjo', 'Arabika Sulawesi Karangan Angin', 'Arabika Surambu Pulu Pulu', 'Arabika Tanah Karo', 'Arabika Tolu Batak', 'Arabika Toraja', 'Arabika Toraja Bolokan', 'Arabika Wamena', 'Arabika Wanoja Kamojang', 'Liberika Lampung', 'Liberika Sumedang', 'Liberika Temanggung', 'Robusta Bali', 'Robusta Bali Pupuan', 'Robusta Dampit', 'Robusta Flores Manggarai', 'Robusta Gayo', 'Robusta Java GKawi', 'Robusta Lampung', 'Robusta Pagar Alam', 'Robusta Pinogu', 'Robusta Semeru', 'Robusta Sidikalang', 'Robusta Temanggung']

model = None

# Preprocessing sesuai training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # Ekstrak state_dict dari checkpoint dictionary
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Bersihkan prefix 'module.' jika ada (sisa DataParallel)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(clean_state_dict)
        model.to(DEVICE)
        model.eval()
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        yield
    finally:
        model = None

app = FastAPI(
    title="Biji Hitam Coffee Classification API",
    description="API untuk klasifikasi jenis biji kopi berdasarkan gambar menggunakan deep learning model ConvNeXt",
    version="1.0.0",
    contact={
        "name": "Coffee Classification Team",
        "url": "https://github.com/yourusername/biji-hitam-backend",
    },
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint untuk health check.
    
    Returns:
        dict: Status API dan informasi model
    """
    return {
        "status": "active",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES
    }

@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Klasifikasi Jenis Biji Kopi",
    responses={
        200: {
            "description": "Prediksi berhasil",
            "content": {
                "application/json": {
                    "example": {
                        "class_name": "Arabika Aceh Gayo",
                        "confidence": 95.32,
                        "index": 0
                    }
                }
            },
        },
        400: {"description": "File bukan gambar atau format tidak valid"},
        413: {"description": "Ukuran file melebihi batas maksimal (2MB)"},
        500: {"description": "Error saat memproses gambar"},
        503: {"description": "Model belum ter-load"},
    }
)
async def predict(
    file: UploadFile = File(..., description="File gambar biji kopi (JPG, PNG, WebP, dst)")
):
    """
    Memprediksi jenis biji kopi dari gambar yang diunggah.
    
    **Parameter:**
    - **file**: File gambar dalam format image/* (maksimal 10MB)
    
    **Return:**
    - **class_name**: Nama jenis biji kopi yang diprediksi
    - **confidence**: Tingkat kepercayaan prediksi dalam persen (0-100)
    - **index**: Index dari kelas yang diprediksi (0-53)
    
    **Total Kelas Kopi:**
    - Arabika: 37 varian
    - Liberika: 3 varian
    - Robusta: 14 varian
    
    **Contoh Response:**
    ```json
    {
        "class_name": "Arabika Aceh Gayo",
        "confidence": 95.32,
        "index": 0
    }
    ```
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        content = await file.read()
        
        # Validasi ukuran file
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE / (1024*1024):.0f}MB")
        
        image = Image.open(io.BytesIO(content)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probs, 1)

        idx = top_idx.item()
        return {
            "class_name": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx),
            "confidence": round(top_prob.item() * 100, 2),
            "index": idx
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))