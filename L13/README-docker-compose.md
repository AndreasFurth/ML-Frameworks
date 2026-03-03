# Docker Compose: three-component example

Example stack: **frontend** (nginx) → **backend** (FastAPI) → **db** (PostgreSQL).

## Prerequisites

- Docker and Docker Compose installed
- For the backend to start: `app.py` (and optionally `model.joblib`) in L13, as in Task 1 of the assignment

## Run

From the `L13` directory:

```bash
docker compose up --build
```

- **Frontend:** http://localhost:3000  
- **Backend API:** http://localhost:8000 (docs at http://localhost:8000/docs)  
- **PostgreSQL:** localhost:5432, user `app`, password `appsecret`, database `appdb`

## Services

| Service   | Image/Build        | Port |
|----------|--------------------|------|
| `db`     | postgres:15-alpine | 5432 |
| `backend`| Dockerfile (.)     | 8000 |
| `frontend` | frontend/Dockerfile | 3000 → 80 |

Backend gets `DATABASE_URL=postgresql://app:appsecret@db:5432/appdb`; you can use it in your FastAPI app when you add DB access.
