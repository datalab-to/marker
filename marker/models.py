import os
import tempfile
from pathlib import Path

from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor

from marker.ocr import IndraRecognitionPredictor


def _ensure_surya_cache_writable() -> None:
    """Redirect Surya's server lock/log directory if ~/.cache is not writable (e.g. read-only HOME containers)."""
    default_dir = Path(os.path.expanduser("~/.cache/datalab/surya"))
    try:
        default_dir.mkdir(parents=True, exist_ok=True)
        probe = default_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        return
    except OSError:
        pass

    fallback_base = Path(
        os.environ.get("XDG_CACHE_HOME") or tempfile.gettempdir()
    )
    fallback_dir = fallback_base / "datalab" / "surya"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    try:
        import surya.inference.backends.spawn as surya_spawn

        surya_spawn._cache_dir = lambda: fallback_dir
        orig_attach_or_spawn = surya_spawn.attach_or_spawn

        def _patched_attach_or_spawn(*args, **kwargs):
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = str(
                fallback_base.parent
                if fallback_base.name == ".cache"
                else fallback_base
            )
            try:
                return orig_attach_or_spawn(*args, **kwargs)
            finally:
                if old_home is not None:
                    os.environ["HOME"] = old_home

        surya_spawn.attach_or_spawn = _patched_attach_or_spawn
        try:
            import surya.inference.backends.llamacpp as surya_llamacpp

            surya_llamacpp.attach_or_spawn = _patched_attach_or_spawn
        except Exception:
            pass
    except Exception:
        pass


_ensure_surya_cache_writable()


def create_model_dict(
    device=None,
    dtype=None,
    attention_implementation: str | None = None,
    inference_manager: SuryaInferenceManager | None = None,
    inference_backend: str | None = None,
    load_indra: bool = True,
    no_server: bool = False,
) -> dict:
    """Build the minimal predictor set marker uses across its four modes:
      - ``no_ocr`` (or ``disable_ocr=True``): embedded PDF text layer +
        lightweight in-process Indra ONNX layout & reading-order head.
      - ``fast`` / ``indra``: in-process Indra line-level ONNX OCR for any
        garbled/missing blocks or scanned pages (no external servers).
      - ``balanced``: smart routing — in-process Indra layout + embedded PDF
        text when clean, with Surya VLM OCR (``recognition_model``) for
        tables, math, and scanned/garbled pages.
      - ``vllm`` (or ``surya`` / ``accurate``): pure VLM mode — runs full-page
        Surya VLM OCR on every page, skipping pdftext and Indra entirely.
    """
    _ensure_surya_cache_writable()
    manager = inference_manager or SuryaInferenceManager(method=inference_backend)
    indra = IndraRecognitionPredictor() if load_indra else None
    rec = RecognitionPredictor(manager)
    return {
        "inference_manager": manager,
        "recognition_model": rec,
        "layout_model": indra,
        "fast_layout_model": indra,
        "ocr_error_model": None,
        **(
            {"indra_model": indra}
            if load_indra
            else {}
        ),
    }


def shutdown_models(model_dict: dict) -> None:
    """Stop the shared inference server if this process spawned it."""
    manager = model_dict.get("inference_manager")
    if manager is not None:
        manager.stop()
