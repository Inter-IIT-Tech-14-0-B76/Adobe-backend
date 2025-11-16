from __future__ import annotations
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum

from pydantic import EmailStr, constr
from sqlmodel import SQLModel, Field
from sqlalchemy import (
    Column,
    String,
    DateTime,
    LargeBinary,
    ForeignKey,
    Index,
    Boolean,
    Text,
    func,
    Enum as SAEnum,
    Column,
    String,
    Integer,
    LargeBinary,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    func
)


class UserStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class PrivacyLevel(str, Enum):
    DEFAULT = "default"
    STRICT = "strict"
    CUSTOM = "custom"

class Project(SQLModel, table=True):
    __tablename__ = "project"
    __table_args__ = (
        Index("ix_project_created_at", "created_at"),
    )

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(String(36), primary_key=True, nullable=False),
    )
    name: Optional[str] = Field(default=None, sa_column=Column(String(255), nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    def public_dict(self):
        return {"id": str(self.id), "name": self.name, "created_at": self.created_at.isoformat()}



class User(SQLModel, table=True):
    __tablename__ = "user"
    __table_args__ = (
        Index("ix_user_firebase_uid", "firebase_uid", unique=True),
        Index("ix_user_email", "email"),
        Index("ix_user_created_at", "created_at"),
    )

    # internal DB primary id (UUID string for portability)
    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(String(36), primary_key=True, nullable=False),
    )

    # authoritative Firebase uid (unique)
    firebase_uid: str = Field(
        sa_column=Column(String(128), nullable=False, unique=True)
    )

    email: EmailStr = Field(sa_column=Column(String(320), nullable=False))
    email_verified: bool = Field(
        default=False, sa_column=Column(String(5), nullable=False)
    )

    display_name: Optional[constr(max_length=150)] = Field(
        default=None, sa_column=Column(String(150), nullable=True)
    )

    # encrypted PII blob (AES envelope encrypted) - optional
    pii_encrypted_blob: Optional[bytes] = Field(
        default=None, sa_column=Column(LargeBinary, nullable=True)
    )

    # timestamps (UTC); server_default on DB side helps but we set defaults in app too
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True), nullable=False, server_default=func.now()
        ),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
    )
    last_login_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    status: UserStatus = Field(
        default=UserStatus.ACTIVE,
        sa_column=Column(SAEnum(UserStatus, name="user_status"), nullable=False),
    )
    privacy_level: PrivacyLevel = Field(
        default=PrivacyLevel.DEFAULT,
        sa_column=Column(SAEnum(PrivacyLevel, name="privacy_level"), nullable=False),
    )
    consent_version_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(
            String(36),
            ForeignKey("consentversion.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )  # convenience: return safe public representation (no PII blob)

    def public_dict(self) -> dict:
        return {
            "id": str(self.id),
            "firebase_uid": self.firebase_uid,
            "email": str(self.email),
            "email_verified": self.email_verified,
            "display_name": self.display_name,
            "status": self.status.value,
            "privacy_level": self.privacy_level.value,
            "consent_version_id": str(self.consent_version_id)
            if self.consent_version_id
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat()
            if self.last_login_at
            else None,
        }


# recently added by harsh 

class Image(SQLModel, table=True):
    __tablename__ = "image"
    __table_args__ = (
        # index for uploader lookup
        Index("ix_image_uploader_user_id", "uploader_user_id"),
        # index for creation time
        Index("ix_image_created_at", "created_at"),
        # index for retention expiration checks
        Index("ix_image_retention_expires_at", "retention_expires_at"),
        # Unique index on (project_id, sha256_hash) to dedupe within a project
        Index("ux_image_project_sha256", "project_id", "sha256_hash", unique=True),
    )

    # Primary key as UUID string
    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(String(36), primary_key=True, nullable=False),
    )

    # Foreign keys
    project_id: UUID = Field(
        sa_column=Column(String(36), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    )
    uploader_user_id: UUID = Field(
        sa_column=Column(String(36), ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    )

    # S3 object key (do NOT store full URL)
    object_key: str = Field(sa_column=Column(String(1024), nullable=False))

    # Original filename (optional)
    file_name: Optional[str] = Field(default=None, sa_column=Column(String(1024), nullable=True))

    # MIME type e.g. image/png
    mime_type: Optional[str] = Field(default=None, sa_column=Column(String(128), nullable=True))

    # image dimensions
    width: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    height: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))

    # storage size in bytes
    size_bytes: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))

    # content-address hash (hex string)
    sha256_hash: str = Field(sa_column=Column(String(128), nullable=False))

    # hashing algorithm used (default sha256)
    content_hash_algo: str = Field(default="sha256", sa_column=Column(String(32), nullable=False))

    # optionally store encrypted EXIF JSON blob (AES envelope encrypted)
    exif_metadata_encrypted: Optional[bytes] = Field(
        default=None, sa_column=Column(LargeBinary, nullable=True)
    )

    # S3 version id if object versioning enabled
    s3_version_id: Optional[str] = Field(default=None, sa_column=Column(String(255), nullable=True))

    # soft-delete flag
    is_deleted: bool = Field(default=False, sa_column=Column(Integer, nullable=False, server_default="0"))

    # retention expiry timestamp (for automatic purge workflows)
    retention_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    # derived sensitivity/scan flags (nsfw, face_detected, license_restriction, etc.)
    # JSON is convenient and flexible; you can also map to an enum/set column if needed.
    sensitivity_flags: Optional[Dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    # timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )

    # uploaded_at: when the uploader uploaded the file (could be same as created_at)
    uploaded_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def public_dict(self) -> dict:
        """
        Safe public representation — intentionally omits encrypted EXIF and internal bytes.
        """
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "uploader_user_id": str(self.uploader_user_id) if self.uploader_user_id else None,
            "object_key": self.object_key,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "content_hash_algo": self.content_hash_algo,
            # we include the hash to allow content-address lookups; remove if you want more privacy
            "sha256_hash": self.sha256_hash,
            "s3_version_id": self.s3_version_id,
            "is_deleted": bool(self.is_deleted),
            "retention_expires_at": self.retention_expires_at.isoformat() if self.retention_expires_at else None,
            "sensitivity_flags": self.sensitivity_flags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


## done 


class ConsentVersion(SQLModel, table=True):
    """
    Represents a published version of the user consent policy.

    - Never update existing versions (append-only log)
    - Users reference one version → User.consent_version_id
    """

    __tablename__ = "consentversion"
    __table_args__ = (
        Index("ix_consentversion_version", "version", unique=True),
        Index("ix_consentversion_created_at", "created_at"),
    )

    id: UUID = Field(
        default_factory=uuid4, sa_column=Column(String(36), primary_key=True)
    )

    # human-readable or semantic version string, ex: "2024-01", "v1.3"
    version: str = Field(
        sa_column=Column(String(64), nullable=False, unique=True),
        description="Version identifier for consent policy.",
    )

    # Long text field — supports JSON, markdown, etc.
    content: str = Field(
        sa_column=Column(Text, nullable=False),
        description="Full consent text / JSON / markdown depending on system.",
    )

    # Whether this version is still active for new signups
    is_active: bool = Field(
        default=True,
        sa_column=Column(Boolean, nullable=False, default=True),
        description="If false, this version is deprecated.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )

    def public_dict(self):
        return {
            "id": str(self.id),
            "version": self.version,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "content": self.content,
        }
