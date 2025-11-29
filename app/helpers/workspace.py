import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from fastapi import HTTPException  

load_dotenv()

logger = logging.getLogger(__name__)


class WorkspaceType(str, Enum):
    """Enumeration of available workspaces in the photo editing application."""

    BASIC_EDIT = "basic_edit"
    ADVANCED_EDIT = "advanced_edit"
    FILTERS = "filters"
    ADJUSTMENTS = "adjustments"
    TEXT = "text"
    STICKERS = "stickers"
    EFFECTS = "effects"
    RETOUCH = "retouch"
    BACKGROUND_REMOVER = "background_remover"
    OBJECT_REMOVER = "object_remover"
    COLLAGE = "collage"


class WorkspaceDecisionInput(BaseModel):
    """Input model for the workspace decision making process."""

    user_prompt: str
    current_image_metadata: Optional[Dict[str, Any]] = (
        None  # can input multiple images as well
    )
    previous_actions: List[Dict[str, Any]] = []
    available_workspaces: List[str] = [w.value for w in WorkspaceType]


class WorkspaceDecision(BaseModel):
    """Output model for the workspace decision."""

    selected_workspace: str
    confidence: float  # redundant
    reasoning: str
    parameters: Optional[Dict[str, Any]] = None


class WorkspaceDecisionMaker:
    """Makes decisions about which workspace to use based on user input using LLM."""

    MODEL_ID = "gemini-2.0-flash"
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"

    DEFAULT_WORKSPACE = WorkspaceType.BASIC_EDIT

    @classmethod
    def _get_llm_response(cls, prompt: str) -> str:
        """Get response from the LLM API."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("API Key missing")

        url_with_key = f"{cls.API_URL}?key={api_key}"

        headers = {"Content-Type": "application/json"}
        system_prompt = """
        You are a photo editing assistant. Your task is to select the most appropriate workspace based on the user's request.
Available workspaces: basic_edit, advanced_edit, filters, adjustments, text, stickers, effects, retouch, background_remover, object_remover, collage.

Rules:
1. Respond with ONLY the workspace name in lowercase.
2. Do not explain your reasoning.
3. Do not use punctuation."""

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{system_prompt}\n\nUser request: {prompt}\nWorkspace:"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 20,
            },
        }

        try:
            response = requests.post(
                url_with_key, headers=headers, json=payload, timeout=10
            )
            response.raise_for_status()

            response_data = response.json()

            if "candidates" in response_data and response_data["candidates"]:
                result = response_data["candidates"][0]["content"]["parts"][0]["text"]
                result = result.strip().lower()
                return result.split()[0]
            else:
                logger.warning(
                    f"Gemini returned no candidates. Full response: {response_data}"
                )
                return cls.DEFAULT_WORKSPACE.value

        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to get workspace decision from the AI model",
            )

    @classmethod
    def decide_workspace(
        cls, input_data: "WorkspaceDecisionInput"
    ) -> "WorkspaceDecision":
        """Determine the most appropriate workspace using LLM."""
        if not input_data.user_prompt:
            return cls._default_decision("No user prompt provided")

        try:
            selected_workspace = cls._get_llm_response(input_data.user_prompt)

            if selected_workspace not in input_data.available_workspaces:
                logger.warning(f"LLM returned invalid workspace: {selected_workspace}")
                selected_workspace = cls.DEFAULT_WORKSPACE.value
                confidence = 0.5
            else:
                confidence = 0.9

            return WorkspaceDecision(
                selected_workspace=selected_workspace,
                confidence=confidence,
                reasoning=f"Selected {selected_workspace} based on user request",
                parameters={},
            )

        except Exception as e:
            logger.error(f"Error in decide_workspace: {str(e)}")
            return cls._default_decision(str(e))

    @classmethod
    def _default_decision(cls, reason: str) -> "WorkspaceDecision":
        """Helper to return a default decision object on error."""
        return WorkspaceDecision(
            selected_workspace=cls.DEFAULT_WORKSPACE.value,
            confidence=0.0,
            reasoning=f"Default decision due to: {reason}",
            parameters={},
        )


def get_workspace_decision(
    user_prompt: str,
    current_image_metadata: Optional[Dict[str, Any]] = None,
    previous_actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Helper function to get workspace decision with a simpler interface.
    """
    if previous_actions is None:
        previous_actions = []

    decision_input = WorkspaceDecisionInput(
        user_prompt=user_prompt,
        current_image_metadata=current_image_metadata or {},
        previous_actions=previous_actions,
    )

    try:
        decision = WorkspaceDecisionMaker.decide_workspace(decision_input)
        return decision.dict()
    except Exception as e:
        logger.error(f"Error determining workspace: {str(e)}", exc_info=True)
        return {
            "selected_workspace": WorkspaceType.BASIC_EDIT.value,
            "confidence": 0.0,
            "reasoning": f"Error processing request: {str(e)}",
            "parameters": None,
        }
