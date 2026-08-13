# Potato Disease Classifier — Backend (Vercel)

FastAPI + TFLite inference for **Early Blight**, **Late Blight**, and **Healthy**.

## Local run

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional local fallback if LiteRT fails on Windows:
# pip install tensorflow
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Deploy on Vercel (this folder = repo root)

1. Push **only this `backend` folder** as its own GitHub repo.
2. Vercel → import that repo (Framework: Other / FastAPI auto-detect).
3. Add env var (optional if using AetherLeaf URL already allowed in code):
   - `FRONTEND_ORIGIN` = `https://aetherleaf.vercel.app`
4. Deploy.

Entrypoint is `app/main.py` (`app = FastAPI()`). Do **not** add catch-all rewrites to `api/index` — that breaks `/predict`.

### Endpoints

- `GET /`
- `GET /health`
- `POST /predict` (multipart field `file`)

## Notes

- Model file: `models/potato_disease.tflite` (must be committed).
- Production uses `ai-edge-litert` (not full TensorFlow).
