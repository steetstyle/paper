"""Fetching models, including the ``pytorch_model.bin`` -> safetensors step.

Why this module exists: ``transformers`` 4.5x refuses to load a checkpoint that
only ships ``pytorch_model.bin`` when torch is older than 2.6 (the
``torch.load`` CVE-2025-32434 mitigation). Several small checkpoints people
actually want - ``Hello-SimpleAI/chatgpt-detector-roberta``,
``ytu-ce-cosmos/turkish-gpt2`` - are still published that way, so on torch 2.5
they fail with a wall of text and no model.

The fix is safe and local: read the weights with ``weights_only=True`` (which
torch 2.5 does allow) and re-serialise them as safetensors inside the snapshot
directory. Nothing is upgraded, nothing is downloaded twice, and the original
file is left untouched.
"""

from __future__ import annotations

from pathlib import Path

from checker_app.logging import get_logger

__all__ = ["ensure_model", "find_snapshot", "convert_bin_to_safetensors"]

logger = get_logger("checker.models")

_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")
_BIN_NAMES = ("pytorch_model.bin",)


def find_snapshot(model_id: str, cache_dir: str | None = None) -> Path | None:
    """Local snapshot directory of ``model_id``, if it has been downloaded."""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError:  # pragma: no cover - depends on the install
        return None
    patterns = ["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"]
    try:
        # Already on disk? Then there is nothing to fetch, and skipping the call
        # keeps runs fast and offline-clean.
        return Path(
            snapshot_download(
                model_id, cache_dir=cache_dir, allow_patterns=patterns, local_files_only=True
            )
        )
    except Exception:
        pass
    try:
        return Path(snapshot_download(model_id, cache_dir=cache_dir, allow_patterns=patterns))
    except Exception as exc:  # pragma: no cover - network/offline dependent
        first = exc
    # Some repos (distilgpt2 among them) 404 on the xet read-token endpoint.
    # Retrying with xet disabled is cheap and unblocks them.
    try:
        import os  # noqa: PLC0415

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        return Path(
            snapshot_download(model_id, cache_dir=cache_dir, allow_patterns=patterns)
        )
    except Exception as exc:  # pragma: no cover - network/offline dependent
        logger.warning("model indirilemedi %s: %s (xetsiz deneme: %s)", model_id, first, exc)
        return None


def ensure_model(model_id: str, cache_dir: str | None = None) -> str | None:
    """Return a local path to a loadable model, converting ``.bin`` if needed.

    Returns ``None`` when the model cannot be fetched at all, so callers can
    degrade instead of raising.
    """
    snapshot = find_snapshot(model_id, cache_dir)
    if snapshot is None:
        return None
    if any((snapshot / name).exists() for name in _WEIGHT_FILES):
        return str(snapshot)
    for name in _BIN_NAMES:
        if (snapshot / name).exists():
            try:
                convert_bin_to_safetensors(snapshot / name, snapshot / "model.safetensors")
            except Exception as exc:  # pragma: no cover - depends on the checkpoint
                logger.warning("%s dönüştürülemedi: %s", model_id, exc)
                return str(snapshot)
            logger.info("%s safetensors'a dönüştürüldü", model_id)
            return str(snapshot)
    return str(snapshot)


def convert_bin_to_safetensors(source: Path, target: Path) -> None:
    """Rewrite a ``.bin`` checkpoint as safetensors using weights-only loading."""
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"beklenmeyen ağırlık yapısı: {type(state).__name__}")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    tensors = {
        key: value.contiguous() for key, value in state.items() if hasattr(value, "contiguous")
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target))
