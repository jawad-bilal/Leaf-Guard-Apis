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
2. [Vercel](https://vercel.com) → Add New Project → import that repo.
3. Framework preset: **Other**. Root directory: `.`
4. Add env var:
   - `FRONTEND_ORIGIN` = `https://your-frontend.vercel.app` (no trailing slash)
5. Deploy.

API base URL will look like `https://your-backend.vercel.app`.

### Endpoints

- `GET /health`
- `POST /predict` (multipart field `file`)

## Notes

- Model file: `models/potato_disease.tflite` (must be committed).
- Full TensorFlow is **not** used in production (too large for Vercel). Inference uses `ai-edge-litert`.
- After the frontend is live, set `FRONTEND_ORIGIN` so CORS allows the UI.
