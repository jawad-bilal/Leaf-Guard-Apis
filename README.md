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
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Deploy on Vercel (this folder = repo root)

1. Push this repo to GitHub and import in Vercel.
2. Framework should detect **FastAPI** (entrypoint `main.py` → `app`).
3. Env (optional): `FRONTEND_ORIGIN=https://aetherleaf.vercel.app`
4. Deploy.

### Endpoints

- `GET /`
- `GET /health`
- `POST /predict` (multipart field `file`)

Model file `models/potato_disease.tflite` must stay in the repo.
