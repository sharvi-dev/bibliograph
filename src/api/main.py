from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
import src.api.state as state


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield
    state.clear()


app = FastAPI(title="BiblioGraph", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": state.is_loaded()}
