from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import HTTPException, Request, status
from firebase_admin import auth as firebase_auth
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.utils.models import PrivacyLevel, User
from sprint import sprint


async def verify_firebase_token(request: Request) -> Dict:
    """
    FastAPI dependency: verifies the Authorization header (Bearer <id_token>)
    and returns the decoded Firebase token payload.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization bearer token",
        )
    id_token = auth_header.split("Bearer ")[1].strip()
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token verification failed"
        )


async def upsert_user_from_token(
    token_payload: Dict,
    session: AsyncSession,
    *,
    set_last_login: bool = True,
    default_privacy: PrivacyLevel = PrivacyLevel.DEFAULT,
    display_name_override: Optional[str] = None,
) -> User:
    """
    Find or create (upsert) a User row based on firebase uid inside token_payload.
    Mirrors minimal safe fields: email, email_verified, display_name.
    Returns the SQLModel User instance (attached to session).
    """
    uid = token_payload["uid"]
    stmt = select(User).where(User.firebase_uid == uid)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    mirrored_email = token_payload.get("email")
    email_verified = bool(token_payload.get("email_verified", False))
    display_name = (
        display_name_override
        or token_payload.get("name")
        or token_payload.get("displayName")
    )

    sprint(f"upsert: display_name_override = {display_name_override}", "d")
    sprint(f"upsert: token name = {token_payload.get('name')}", "d")
    sprint(f"upsert: token displayName = {token_payload.get('displayName')}", "d")
    sprint(f"upsert: final display_name = {display_name}", "d")

    if user is None:
        user = User(
            firebase_uid=uid,
            email=mirrored_email or "",
            email_verified=email_verified,
            display_name=display_name,
            created_at=now,
            updated_at=now,
            last_login_at=now if set_last_login else None,
            privacy_level=default_privacy,
            status=user_status_active(),
        )
        session.add(user)
        try:
            await session.commit()
            await session.refresh(user)
        except IntegrityError:
            await session.rollback()
            result = await session.execute(stmt)
            user = result.scalar_one()
    else:
        updated = False
        if mirrored_email and user.email != mirrored_email:
            user.email = mirrored_email
            updated = True
        if user.email_verified != email_verified:
            user.email_verified = email_verified
            updated = True
        if display_name and user.display_name != display_name:
            user.display_name = display_name
            updated = True
        if set_last_login:
            user.last_login_at = now
            updated = True
        if updated:
            session.add(user)
            await session.commit()
            await session.refresh(user)
    return user


def user_status_active():
    from app.utils.models import UserStatus
    return UserStatus.ACTIVE
