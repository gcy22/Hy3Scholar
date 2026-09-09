from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .schemas import AdversarialReport, PipelineResult, Workspace
from .service import Hy3ScholarService


app = FastAPI(
    title="Hy3Scholar API",
    version="0.1.0",
    description="Hy3 驱动的证据增强文献综述生成与可信评估",
)
settings = Settings()
service = Hy3ScholarService(settings)


class RunRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=12, ge=3, le=30)
    refine: bool = True


class EvaluateRequest(BaseModel):
    question: str = Field(min_length=3)
    markdown: str = Field(min_length=20)


class AdversarialRequest(BaseModel):
    question: str = Field(min_length=3)
    markdown: str = Field(min_length=20)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": settings.model,
        "api_key_configured": bool(settings.api_key),
    }


@app.post("/workspaces", response_model=Workspace)
def create_workspace(files: list[UploadFile] = File(...)) -> Workspace:
    if not files:
        raise HTTPException(400, "至少上传一个 PDF")
    try:
        with tempfile.TemporaryDirectory(prefix="hy3scholar-upload-") as tmp:
            paths: list[Path] = []
            for index, upload in enumerate(files, 1):
                if not upload.filename or not upload.filename.lower().endswith(".pdf"):
                    raise HTTPException(400, f"仅支持 PDF：{upload.filename}")
                original_name = Path(upload.filename).name
                path = Path(tmp) / f"{index:03d}-{original_name}"
                with path.open("wb") as handle:
                    shutil.copyfileobj(upload.file, handle)
                if path.stat().st_size > 50 * 1024 * 1024:
                    raise HTTPException(413, f"文件超过 50 MB：{upload.filename}")
                paths.append(path)
            return service.create_workspace(paths)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/workspaces/{workspace_id}/run", response_model=PipelineResult)
def run_pipeline(workspace_id: str, request: RunRequest) -> PipelineResult:
    try:
        return service.run(
            workspace_id,
            request.question,
            top_k=request.top_k,
            refine=request.refine,
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/workspaces/{workspace_id}/evaluate", response_model=PipelineResult)
def evaluate_review(workspace_id: str, request: EvaluateRequest) -> PipelineResult:
    try:
        return service.evaluate(workspace_id, request.question, request.markdown)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post(
    "/workspaces/{workspace_id}/adversarial-test",
    response_model=AdversarialReport,
)
def adversarial_test(
    workspace_id: str, request: AdversarialRequest
) -> AdversarialReport:
    try:
        return service.adversarial_test(
            workspace_id, request.markdown, request.question
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
