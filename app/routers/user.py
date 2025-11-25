from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, constr
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.helpers.auth import upsert_user_from_token, verify_firebase_token
from app.utils.db import async_session
from app.utils.models import PrivacyLevel, User

user_router = APIRouter(prefix="/auth", tags=["auth"])


class SignupRequest(BaseModel):
    display_name: Optional[constr(max_length=150)] = None
    privacy_level: Optional[PrivacyLevel] = None
    consent_version_id: Optional[str] = None


class LoginRequest(BaseModel):
    display_name: Optional[constr(max_length=150)] = None


@user_router.post("/login")
async def login_route(
    body: LoginRequest,
    token_payload: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    """
    Login endpoint: verifies token (via dependency) and returns the upserted user.
    - clients should call Firebase Auth on client and pass the ID token in Authorization header.
    """
    user = await upsert_user_from_token(
        token_payload, session, display_name_override=body.display_name
    )
    return user.public_dict()


@user_router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup_route(
    body: SignupRequest,
    token_payload: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    """
    Signup endpoint: create user (if not exists) and apply onboarding fields.
    If user exists, will update allowed fields (display_name, privacy_level, consent_version_id).
    """
    user = await upsert_user_from_token(
        token_payload,
        session,
        set_last_login=True,
        display_name_override=body.display_name,
    )

    updated = False

    if body.privacy_level and user.privacy_level != body.privacy_level:
        user.privacy_level = body.privacy_level
        updated = True

    if body.consent_version_id:
        user.consent_version_id = body.consent_version_id
        updated = True

    if updated:
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user.public_dict()


@user_router.post("/logout")
async def logout_route(
    token_payload: dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    """
    Logout endpoint: updates last_login_at for the user.
    """
    uid = token_payload["uid"]
    stmt = select(User).where(User.firebase_uid == uid)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        user.last_login_at = datetime.now(timezone.utc)
        session.add(user)
        await session.commit()

    return {"message": "Logged out successfully"}
