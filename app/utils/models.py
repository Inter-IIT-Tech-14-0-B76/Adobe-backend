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
            String(36),
            ForeignKey("version_history.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    image_ids: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, server_default="[]"),
    )

    prompt: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    output_logs: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    feedback: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    project: "Project" = Relationship(
        back_populates="versions",
        sa_relationship_kwargs={"foreign_keys": "[VersionHistory.project_id]"},
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
            String(36),
            ForeignKey("version_history.id", ondelete="SET NULL"),
            nullable=True,
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
            "parent_image_id": str(self.parent_image_id)
            if self.parent_image_id
            else None,
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


class Project3D(SQLModel, table=True):
    """
    Model for 3D project visualization with conversational workflow support.

    Supports a conversational workflow where:
    1. User uploads 4 images -> initial video is generated
    2. User can send prompts (with optional new images) to update the video
    3. GLB file can be generated at any time using latest state

    The `id` serves as the persistent identifier across all operations:
    - Initial image upload
    - Subsequent prompt updates
    - GLB generation requests

    Storage locations:
    - GLB files: /workspace/backend/glb/
    - Demo videos: /workspace/backend/demo_videos/
    """

    __tablename__ = "project_3d"
    __table_args__ = (Index("ix_project_3d_p_id", "p_id"),)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        sa_column=Column(String(36), primary_key=True),
    )

    p_id: Optional[str] = Field(
        default=None,
        sa_column=Column(
            String(36), ForeignKey("project.id", ondelete="CASCADE"), nullable=True
        ),
        description="Optional reference to a parent project",
    )

    # Four input images provided by the user (S3 object keys or local paths)
    image_1: Optional[str] = Field(
        default=None,
        sa_column=Column(String(1024), nullable=True),
        description="First input image path/key",
    )
    image_2: Optional[str] = Field(
        default=None,
        sa_column=Column(String(1024), nullable=True),
        description="Second input image path/key",
    )
    image_3: Optional[str] = Field(
        default=None,
        sa_column=Column(String(1024), nullable=True),
        description="Third input image path/key",
    )
    image_4: Optional[str] = Field(
        default=None,
        sa_column=Column(String(1024), nullable=True),
        description="Fourth input image path/key",
    )

    # GLB file path stored on disk at /workspace/backend/glb/
    glb_file_path: Optional[str] = Field(
        default=None,
        sa_column=Column(String(1024), nullable=True),
        description="Path to the generated GLB file on disk",
    )

    # Demo video path stored on disk at /workspace/backend/demo_videos/
    demo_video_path: Optional[str] = Field(
        default=None,
        sa_column=Column(String(1024), nullable=True),
        description="Path to the latest demo video (MP4) on disk",
    )

    # Conversational workflow fields
    latest_prompt: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="The most recent prompt used for video generation",
    )

    prompt_history: List[Dict] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, server_default="[]"),
        description="History of prompts and their associated video paths",
    )

    video_history: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, server_default="[]"),
        description="List of all generated video paths for this project",
    )

    generation_count: int = Field(
        default=0,
        sa_column=Column(String(10), nullable=False, server_default="0"),
        description="Number of video generations performed",
    )

    # Job key from latest /infer call (needed for /convert to GLB)
    latest_job_key: Optional[str] = Field(
        default=None,
        sa_column=Column(String(256), nullable=True),
        description="Job key from latest inference (used for GLB conversion)",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
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

    def get_images(self) -> List[str]:
        """Return list of non-null image paths."""
        return [
            img
            for img in [self.image_1, self.image_2, self.image_3, self.image_4]
            if img is not None
        ]

    def set_images(self, images: List[str]) -> None:
        """Set image paths from a list (up to 4 images)."""
        self.image_1 = images[0] if len(images) > 0 else None
        self.image_2 = images[1] if len(images) > 1 else None
        self.image_3 = images[2] if len(images) > 2 else None
        self.image_4 = images[3] if len(images) > 3 else None

    def add_to_history(
        self,
        prompt: Optional[str],
        video_path: str,
        job_key: Optional[str] = None,
    ) -> None:
        """Add a generation to the prompt and video history."""
        history_entry = {
            "prompt": prompt,
            "video_path": video_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_number": self.generation_count + 1,
            "job_key": job_key,
        }
        self.prompt_history = self.prompt_history + [history_entry]
        self.video_history = self.video_history + [video_path]
        self.generation_count = int(self.generation_count) + 1
        self.latest_prompt = prompt
        self.demo_video_path = video_path
        if job_key:
            self.latest_job_key = job_key

    def public_dict(self) -> dict:
        """Return a dictionary representation safe for API responses."""
        return {
            "id": str(self.id),
            "p_id": str(self.p_id) if self.p_id else None,
            "images": self.get_images(),
            "image_1": self.image_1,
            "image_2": self.image_2,
            "image_3": self.image_3,
            "image_4": self.image_4,
            "glb_file_path": self.glb_file_path,
            "demo_video_path": self.demo_video_path,
            "latest_prompt": self.latest_prompt,
            "latest_job_key": self.latest_job_key,
            "prompt_history": self.prompt_history,
            "video_history": self.video_history,
            "generation_count": int(self.generation_count),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
