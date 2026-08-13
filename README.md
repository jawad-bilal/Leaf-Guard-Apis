# Potato Disease Classifier — Backend (Vercel)

## Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Vercel

- Entrypoint: `app/main.py` (`app = FastAPI()`)
- After pushing, open Vercel → **Deployments** and confirm the latest commit is live
- If still 404, click **Redeploy** (clear cache)
- Env: `FRONTEND_ORIGIN=https://aetherleaf.vercel.app`

Endpoints: `GET /`, `GET /health`, `POST /predict`
