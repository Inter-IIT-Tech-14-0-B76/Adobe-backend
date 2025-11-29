from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import uuid4

from pydantic import EmailStr
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    pass

# --- ENUMS ---

class UserStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class PrivacyLevel(str, Enum):
    DEFAULT = "default"
    STRICT = "strict"
    CUSTOM = "custom"


class ImageActionType(str, Enum):
    UPLOAD = "upload"
    GENERATE = "generate"
    EDIT = "edit"


# --- MODELS ---

class ConsentVersion(SQLModel, table=True):
    __tablename__ = "consentversion"
    __table_args__ = (Index("ix_consentversion_version", "version", unique=True),)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        sa_column=Column(String(36), primary_key=True),
    )
    version: str = Field(sa_column=Column(String(64), nullable=False, unique=True))
    content: str = Field(sa_column=Column(Text, nullable=False))
    is_active: bool = Field(
        default=True, sa_column=Column(Boolean, nullable=False, default=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=False
        ),
    )


class User(SQLModel, table=True):
    __tablename__ = "user"
    __table_args__ = (
        Index("ix_user_firebase_uid", "firebase_uid", unique=True),
        Index("ix_user_email", "email"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        sa_column=Column(String(36), primary_key=True, nullable=False),
    )
    firebase_uid: str = Field(
        sa_column=Column(String(128), nullable=False, unique=True)
    )
    email: EmailStr = Field(sa_column=Column(String(320), nullable=False))
    email_verified: bool = Field(
        default=False, sa_column=Column(Boolean, nullable=False, default=False)
    )
    display_name: Optional[str] = Field(
        default=None, sa_column=Column(String(150), nullable=True)
    )
    pii_encrypted_blob: Optional[bytes] = Field(
        default=None, sa_column=Column(LargeBinary, nullable=True)
    )
    status: UserStatus = Field(
        default=UserStatus.ACTIVE,
        sa_column=Column(SAEnum(UserStatus, name="user_status"), nullable=False),
    )
    consent_version_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String(36),
            ForeignKey("consentversion.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
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
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    privacy_level: PrivacyLevel = Field(
        default=PrivacyLevel.DEFAULT,
        sa_column=Column(
            SAEnum(PrivacyLevel, name="privacy_level"),
            nullable=False,
        ),
    )

    projects: List["Project"] = Relationship(back_populates="user")

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "firebase_uid": self.firebase_uid,
            "email": str(self.email),
            "display_name": self.display_name,
            "status": self.status.value,
            "privacy_level": self.privacy_level.value
            if hasattr(self, "privacy_level")
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat()
            if self.last_login_at
            else None,
        }


class VersionHistory(SQLModel, table=True):
    __tablename__ = "version_history"
    __table_args__ = (
        Index("ix_version_project_id", "project_id"),
        Index("ix_version_parent_id", "parent_id"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        sa_column=Column(String(36), primary_key=True),
    )

    project_id: str = Field(
        sa_column=Column(
            String(36), ForeignKey("project.id", ondelete="CASCADE"), nullable=False
        )
    )
    parent_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String(36), ForeignKey("version_history.id", ondelete="SET NULL"), nullable=True
        ),
    )
    image_ids: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, server_default="[]")
    )

    prompt: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    
    output_logs: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    
    feedback: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    project: "Project" = Relationship(
        back_populates="versions",
        sa_relationship_kwargs={
            "foreign_keys": "[VersionHistory.project_id]"
        }
    )
    

    parent: Optional["VersionHistory"] = Relationship(
        back_populates="children",
        sa_relationship_kwargs={
            "remote_side": "[VersionHistory.id]",
        },
    )
    
    children: List["VersionHistory"] = Relationship(
        back_populates="parent",
    )

    def public_dict(self) -> dict:
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "image_ids": self.image_ids, 
            "prompt": self.prompt,
            "output_logs": self.output_logs,
            "feedback": self.feedback,
            "created_at": self.created_at.isoformat(),
        }


class Project(SQLModel, table=True):
    __tablename__ = "project"
    __table_args__ = (
        Index("ix_project_user_id", "user_id"),
        Index("ix_project_current_version_id", "current_version_id"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        sa_column=Column(String(36), primary_key=True),
    )
    name: str = Field(
        default="Untitled Project", sa_column=Column(String(255), nullable=False)
    )

    user_id: str = Field(
        sa_column=Column(
            String(36), ForeignKey("user.id", ondelete="CASCADE"), nullable=False
        )
    )

    current_version_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String(36), ForeignKey("version_history.id", ondelete="SET NULL"), nullable=True
        ),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    user: User = Relationship(back_populates="projects")

    versions: List["VersionHistory"] = Relationship(
        back_populates="project",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "foreign_keys": "[VersionHistory.project_id]",
        },
    )

    current_version: Optional["VersionHistory"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Project.current_version_id]",
            "post_update": True,
            "uselist": False,
        }
    )

    def public_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "user_id": str(self.user_id),
            "current_version_id": str(self.current_version_id)
            if self.current_version_id
            else None,
            "created_at": self.created_at.isoformat(),
        }


class Image(SQLModel, table=True):
    __tablename__ = "image"
    __table_args__ = (
        Index("ix_image_parent_image_id", "parent_image_id"),
        Index("ix_image_sha256_hash", "sha256_hash"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        sa_column=Column(String(36), primary_key=True),
    )

    
    project_id: str = Field(
        sa_column=Column(
            String(36), ForeignKey("project.id", ondelete="CASCADE"), nullable=False
        )
    )

    parent_image_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String(36), ForeignKey("image.id", ondelete="SET NULL"), nullable=True
        ),
    )

    action_type: ImageActionType = Field(
        default=ImageActionType.UPLOAD,
        sa_column=Column(String(50), nullable=False),
    )

    is_virtual: bool = Field(
        default=False, sa_column=Column(Boolean, nullable=False, default=False)
    )

    object_key: str = Field(sa_column=Column(String(1024), nullable=False))

    file_name: Optional[str] = Field(
        default=None, sa_column=Column(String(512), nullable=True)
    )

    mime_type: Optional[str] = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )

    sha256_hash: Optional[str] = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )

    transformations: Dict = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, server_default="{}"),
    )

    generation_params: Dict = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False, server_default="{}"),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )


    parent_image: Optional["Image"] = Relationship(
        back_populates="children_images",
        sa_relationship_kwargs={
            "remote_side": "[Image.id]",
            "foreign_keys": "[Image.parent_image_id]",
        },
    )

    children_images: List["Image"] = Relationship(
        back_populates="parent_image",
        sa_relationship_kwargs={
            "foreign_keys": "[Image.parent_image_id]",
        },
    )

    def public_dict(self) -> dict:
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "parent_image_id": str(self.parent_image_id) if self.parent_image_id else None,
            "action_type": self.action_type.value
            if isinstance(self.action_type, ImageActionType)
            else self.action_type,
            "is_virtual": self.is_virtual,
            "object_key": self.object_key,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "sha256_hash": self.sha256_hash,
            "transformations": self.transformations,
            "generation_params": self.generation_params,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }