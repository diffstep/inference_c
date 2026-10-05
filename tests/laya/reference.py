#!/usr/bin/env python3
"""Official Laya GPU reference workload for SDK parity and latency comparisons."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_MODEL_DIR = (
    "/Users/dengtao2nd/.cache/huggingface/hub/models--convaiinnovations--laya/"
    "snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/multilingual"
)

STATE = (
    "客户提交工单：产线视觉检测站从今天上午开始间歇性漏检，重启后暂时恢复。"
    "现场已经尝试清洁镜头，问题仍然出现；当前班次每小时约有两批产品需要人工复检。"
)

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "选择最适合负责处理该问题的部门。",
        "criteria": {
            "maintenance": "设备维修与产线维护",
            "software": "视觉检测软件与算法",
            "quality": "质量管理与产品判定",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "评估这个问题对当前生产的紧急程度。",
        "criteria": ["不紧急", "需要尽快处理", "正在阻塞生产"],
    },
    "escalate": {
        "type": "noul",
        "instructions": "是否需要立即升级给值班负责人？",
    },
}


def sync_device(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> int:
    model_dir = Path(os.environ.get("LAYA_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
    requested = os.environ.get("LAYA_DEVICE", "auto").lower()
    runs = int(os.environ.get("LAYA_RUNS", "5"))
    warmups = int(os.environ.get("LAYA_WARMUP_RUNS", "2"))
    if runs < 1 or warmups < 0:
        raise SystemExit("LAYA_RUNS must be >= 1 and LAYA_WARMUP_RUNS must be >= 0")

    try:
        import torch
        from laya import Agent
    except ImportError as exc:
        raise SystemExit(
            "Missing reference dependencies. Install with `python3 -m pip install laya`."
        ) from exc

    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            raise SystemExit("No CUDA or Metal (MPS) GPU is available; CPU inference is disabled.")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("LAYA_DEVICE=cuda requested, but CUDA is unavailable.")
        device = "cuda"
    elif requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise SystemExit("LAYA_DEVICE=mps requested, but Metal MPS is unavailable.")
        device = "mps"
    else:
        raise SystemExit("LAYA_DEVICE must be auto, cuda, or mps; CPU is intentionally unsupported.")

    if not model_dir.is_dir():
        raise SystemExit(f"model directory not found: {model_dir}")

    # Laya's Agent has CPU fallback behavior in some failure cases. Explicitly verify its
    # resolved device so this GPU-only example can never report a CPU run as a GPU result.
    agent = Agent(str(model_dir), device=device)
    actual_device = str(agent.device)
    if actual_device.split(":", 1)[0] != device:
        raise SystemExit(f"requested {device}, but Laya resolved to {actual_device}; refusing CPU fallback")

    def predict() -> object:
        result = agent.predict(STATE, QUESTIONS)
        actual = str(agent.device)
        if actual.split(":", 1)[0] != device:
            raise RuntimeError(
                f"Laya changed devices during inference ({device} -> {actual}); "
                "refusing to report a CPU fallback as GPU performance"
            )
        sync_device(torch, device)
        return result

    for _ in range(warmups):
        predict()

    latencies = []
    result = None
    for _ in range(runs):
        sync_device(torch, device)
        start = time.perf_counter()
        result = predict()
        latencies.append((time.perf_counter() - start) * 1000.0)

    report_tool = os.environ.get("LAYA_REPORT_TOOL")
    if not report_tool or not Path(report_tool).is_file():
        raise SystemExit("LAYA_REPORT_TOOL is missing; build it with `make -C tests/laya`")
    report = subprocess.run(
        [report_tool, "laya_typed_decisions_reference", device, str(warmups),
         str(len(QUESTIONS)), *(f"{latency:.9f}" for latency in latencies)],
        check=True, capture_output=True, text=True,
    )
    print(report.stdout)
    print(f"  model: {model_dir}")
    print("  engine: official Laya / PyTorch reference (not the dsinfra SDK executor)")
    print("  result:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
