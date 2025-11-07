from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pymongo.collection import Collection
from typing import Dict, List, Optional
from app.db import users_collection
from app.helpers import verify_firebase_token, get_uid

user_router = APIRouter()
