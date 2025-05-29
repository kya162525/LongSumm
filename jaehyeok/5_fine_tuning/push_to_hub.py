"""공통 HF Hub 업로드 유틸리티

이 모듈 하나만 import 하면, 개별 스크립트마다 복잡한 Hub 업로드 로직을
다시 짤 필요 없이 다음 두 인터페이스만 사용하면 됩니다.

1. push_folder()          : (폴더 전체) 한 번만 업로드할 때 사용
2. HFUploadCallback (PT)  : Trainer 단계별로 주기적으로 자동 업로드할 때 사용

LoRA 어댑터만 업로드하고 싶을 때는 adapter_only=True 옵션을 주면 됩니다.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import logging
from typing import Optional

from huggingface_hub import HfApi, login
from transformers import TrainerCallback

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _ensure_login(token: Optional[str] = None) -> tuple[HfApi, str]:
    """Hugging Face Hub 로그인 후 HfApi 인스턴스를 반환"""
    if token is None:
        token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN 환경변수나 인자를 통해 토큰을 제공해야 합니다.")

    # 여러 프로세스가 동시에 login() 을 호출해도 idempotent 합니다.
    login(token=token)
    return HfApi(), token


def push_folder(
    *,
    folder_path: str,
    repo_id: str,
    hf_token: Optional[str] = None,
    create_if_missing: bool = True,
    path_in_repo: str | None = None,
) -> None:
    """folder_path 전체를 repo_id(모델 레포)에 업로드한다."""
    api, token = _ensure_login(hf_token)

    if create_if_missing:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        path_in_repo=path_in_repo,
    )
    _logger.info("✅  Successfully pushed %s → %s", folder_path, repo_id)


# ---------------------------------------------------------------------------
# 🤗 Transformers Trainer callback
# ---------------------------------------------------------------------------
class HFUploadCallback(TrainerCallback):
    """Trainer 중간 체크포인트를 HF Hub 로 주기적으로 업로드합니다."""

    def __init__(
        self,
        *,
        repo_id: str,
        push_every_n_steps: int = 100,
        hf_token: Optional[str] = None,
        adapter_only: bool = False,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.push_every_n_steps = push_every_n_steps
        self.adapter_only = adapter_only
        self.api, self.token = _ensure_login(hf_token)

        try:
            self.api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=self.token)
        except KeyError as e:
            # 이미 존재하는 레포이면 무시
            _logger.warning(f"Repo {repo_id} may already exist, skipping creation (caught {e})")
        except Exception as e:
            _logger.error(f"Unexpected error creating repo: {e}")
            raise

    # --------------------------------------------------
    # TrainerCallback hooks
    # --------------------------------------------------
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.push_every_n_steps != 0:
            return

        model = kwargs["model"]
        tokenizer = kwargs.get("tokenizer", None)

        _logger.info("📤  Uploading checkpoint @step %s to %s", state.global_step, self.repo_id)

        tmpdir = tempfile.mkdtemp(prefix="hf_upload_")
        try:
            # LoRA 어댑터만 올리고 싶다면 PEFT 의 save_pretrained()가 알아서
            # adapter weights 만 저장해줍니다.
            model.save_pretrained(tmpdir, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmpdir)

            # checkpoint-XXXX 식으로 하위 폴더에 쌓아 둔다.
            self.api.upload_folder(
                folder_path=tmpdir,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
                path_in_repo=f"checkpoint-{state.global_step}",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def on_train_end(self, args, state, control, **kwargs):
        # 마지막으로 한 번 더 (안전하게) 업로드하고 종료
        if state.is_world_process_zero:
            self.on_step_end(args, state, control, **kwargs)
