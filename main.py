import logging
import os

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.image import image_router
from app.routers.user import user_router
from app.routers.cv import router as cv_router
from app.routers.workspace import workspace_router
from app.routers.style_transfer import router as style_transfer_router
from app.utils.db import init_db

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


app = FastAPI()


@app.on_event("startup")
async def on_startup():
    logger.info("App startup: initializing DB.")
    await init_db()


origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:*",  # For Flutter mobile/desktop
    "http://127.0.0.1:*",  # For Flutter mobile/desktop
]


env_orgs = os.getenv("FRONTEND_URL").split(",")
for origin in env_orgs:
    if origin not in origins:
        origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router)
app.include_router(user_router, prefix="/user")
app.include_router(cv_router)
app.include_router(workspace_router)
app.include_router(style_transfer_router)