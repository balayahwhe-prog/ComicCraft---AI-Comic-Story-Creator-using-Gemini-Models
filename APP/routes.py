from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from .ai.gemini_flash import (
    generate_gemini_text,
    generate_gemini_image,
    generate_huggingface_text,
    generate_huggingface_image,
    generate_pollinations_image,
)

router = APIRouter()


class PromptRequest(BaseModel):
    prompt: str
    provider: str = "gemini"
    type: str = "text"


# =========================================================
# HOME PAGE
# =========================================================

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# =========================================================
# TEXT GENERATION
# =========================================================

@router.post("/generate")
async def generate(request: PromptRequest):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )

    try:

        provider = request.provider.lower()

        # ---------------- GEMINI ----------------
        if provider == "gemini":

            result = generate_gemini_text(
                request.prompt
            )

        # ---------------- HUGGING FACE ----------------
        elif provider in ["huggingface", "hf"]:

            result = generate_huggingface_text(
                request.prompt
            )

        else:

            raise HTTPException(
                status_code=400,
                detail="Invalid provider."
            )

        return {
            "success": True,
            "result": result
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# =========================================================
# IMAGE GENERATION
# =========================================================

@router.post("/generate-image")
async def generate_image(request: PromptRequest):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Image prompt cannot be empty."
        )

    try:

        provider = request.provider.lower()

        # ---------------- GEMINI ----------------
        if provider == "gemini":

            image_bytes = generate_gemini_image(
                request.prompt
            )

        # ---------------- HUGGING FACE ----------------
        elif provider in ["huggingface", "hf"]:

            image_bytes = generate_huggingface_image(
                request.prompt
            )

        # ---------------- POLLINATIONS ----------------
        elif provider == "pollinations":

            image_bytes = generate_pollinations_image(
                request.prompt
            )

        else:

            raise HTTPException(
                status_code=400,
                detail="Invalid provider."
            )

        return Response(
            content=image_bytes,
            media_type="image/png"
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )