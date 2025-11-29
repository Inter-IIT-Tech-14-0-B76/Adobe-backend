from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import select

from app.helpers.auth import verify_firebase_token, upsert_user_from_token
from app.helpers.workspace import get_workspace_decision
from app.utils.db import async_session
from app.utils.models import Project

workspace_router = APIRouter(tags=["Workspace"])

@workspace_router.post(
    "/workspace/decide",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Determine the appropriate workspace for a task",
    description="""
    Analyzes the user's request and determines the most appropriate workspace
    for the given task using an LLM.
    """
)
async def decide_workspace(
    user_prompt: str = Body(..., embed=True, description="User's request or prompt"),
    project_id: Optional[str] = Body(None, description="Optional project ID for context"),
    token_payload: Dict[str, Any] = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session)
) -> Dict[str, Any]:
    """
    Determine the most appropriate workspace based on the user's request.
    
    This endpoint uses an LLM to analyze the user's request and determine
    which workspace would be most appropriate for the task.
    """
    try:
        user = await upsert_user_from_token(token_payload, session, set_last_login=False)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid user")
        
        current_image_metadata = None
        if project_id:
            result = await session.execute(select(Project).where(Project.id == project_id))
            project = result.scalars().first()
            
            if project and project.user_id != user.id:
                raise HTTPException(status_code=403, detail="Not authorized to access this project")
                
            current_image_metadata = {
                "project_id": project_id,
                "has_project": project is not None
            }
        print("Currently here: ", current_image_metadata)
        decision = get_workspace_decision(
            user_prompt=user_prompt,
            current_image_metadata=current_image_metadata
        )
        
        return {
            "success": True,
            "data": decision
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error determining workspace: {str(e)}"
        )

@workspace_router.get(
    "/workspace/available",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get list of available workspaces",
    description="Returns a list of all available workspaces in the system."
)
async def get_available_workspaces(
    token_payload: Dict = Depends(verify_firebase_token)
) -> Dict[str, Any]:
    """
    Get a list of all available workspaces in the system.
    """
    from app.helpers.workspace import WorkspaceType
    
    workspaces = [{
        "id": workspace.value,
        "name": workspace.name.replace("_", " ").title(),
        "description": f"{workspace.name.replace('_', ' ').title()} workspace"
    } for workspace in WorkspaceType]
    
    return {
        "success": True,
        "data": {
            "workspaces": workspaces
        }
    }
