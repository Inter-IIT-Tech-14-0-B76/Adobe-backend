from fastapi import APIRouter, Depends, Query, Path
from fastapi.responses import JSONResponse, Response
from typing import List, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

from app.helpers import verify_firebase_token, get_uid
import json

router = APIRouter()
