import logging, warnings, os


def quiet_startup() -> None:
    # Reduce noisy distributed logs by default; users can override.
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
    # Suppress C++ logger warnings and stack traces from libtorch/c10d during teardown
    # Must be set BEFORE importing torch anywhere.
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    # Fallback for older builds using c10 logger env var name
    os.environ.setdefault("C10_LOG_LEVEL", "ERROR")
    # Hide C++ stack traces for benign teardown exceptions
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")
    # Prefer new var; drop deprecated to avoid warning
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if "NCCL_ASYNC_ERROR_HANDLING" in os.environ:
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    # Quiet NCCL unless there's an actual error
    os.environ.setdefault("NCCL_DEBUG", "ERROR")

    warnings.filterwarnings("ignore", message=r".*predict_dataloader.*num_workers.*")
    warnings.filterwarnings(
        "ignore", message=r".*tensorboardX.*removed as a dependency.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated",
        category=FutureWarning,
        module=r"torch\.cuda",
    )
    # Suppress common, non-fatal PyTorch warnings that spam across ranks
    warnings.filterwarnings(
        "ignore",
        message=r".*TF32 behavior.*",
        category=UserWarning,
        module=r"torch",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*is_fx_tracing will return true.*",
        category=UserWarning,
        module=r"torch\.fx",
    )

    # Route warnings through logging and turn down common noisy loggers
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    logging.getLogger("torch.fx").setLevel(logging.ERROR)


def set_fp32_precision(mode: str | None) -> None:
    """Set FP32 matmul/conv precision using the new per-backend API.

    Accepts legacy values "high"/"highest" (TF32) and "medium" (IEEE).
    Users may also pass "tf32" or "ieee" directly.
    """
    # Import torch lazily so env vars in quiet_startup() take effect before torch loads
    import torch  # local import on purpose

    if mode is None:
        return
    normalized = mode.lower()
    if normalized in {"high", "highest", "tf32"}:
        # Prefer new API over deprecated allow_tf32 / set_float32_matmul_precision
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        # cuDNN conv precision
        if hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
    elif normalized in {"medium", "ieee"}:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        if hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:
        # Fall back to safe default without warning spam
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        if hasattr(torch.backends.cudnn, "conv"):
            torch.backends.cudnn.conv.fp32_precision = "ieee"
