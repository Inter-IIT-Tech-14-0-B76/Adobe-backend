import os
import dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from app.router import router
from app.user_router import user_router

dotenv.load_dotenv()


app = FastAPI()


origins = [
    "http://localhost:3000",
    "http://localhost:8000",
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

app.include_router(router)
app.include_router(user_router, prefix="/user")
