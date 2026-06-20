from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Bibliograph", version="0.1.0")
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
