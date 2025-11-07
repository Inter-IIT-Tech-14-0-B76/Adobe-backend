from fastapi import Request, HTTPException, status, Depends
from firebase_admin import auth as firebase_auth
from typing import Dict, List
from app.bigquery import run_bigquery
from app.db import users_collection


async def verify_firebase_token(request: Request) -> Dict:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token"
        )

    id_token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        print("[Auth Success] Token verified successfully")
        return decoded_token
    except Exception as e:
        print(f"[Auth Error] Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed"
        )


def get_uid(token_payload: Dict = Depends(verify_firebase_token)) -> str:
    return token_payload["uid"]
