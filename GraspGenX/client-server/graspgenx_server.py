#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a GraspGenX ZMQ inference server.

Usage:
    # Serve with default settings (assets at /code/assets):
    python client-server/graspgenx_server.py \\
        --config checkpoints/gen/config.yaml \\
        --assets_dir /code/assets

    # Pre-load a specific gripper:
    python client-server/graspgenx_server.py \\
        --config checkpoints/gen/config.yaml \\
        --assets_dir /code/assets \\
        --default_gripper franka_panda

    # Custom port:
    python client-server/graspgenx_server.py \\
        --config checkpoints/gen/config.yaml \\
        --assets_dir /code/assets \\
        --port 5557
"""

import argparse
import logging
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start a GraspGenX ZMQ inference server (cross-embodiment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the GraspGenX model config YAML (e.g. checkpoints/gen/config.yaml)",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="/code/assets",
        help="Root directory containing x_grippers/ and proc_grippers/ subdirectories (default: /code/assets)",
    )
    parser.add_argument(
        "--default_gripper",
        type=str,
        default=None,
        help="Default gripper to pre-load and use when clients omit gripper_name (e.g. franka_panda)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Address to bind the ZMQ socket (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="Port to bind the ZMQ socket (default: 5556)",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Accelerate the diffusion denoiser with TensorRT (opt-in; needs "
        "the 'tensorrt' extra: `uv sync --extra tensorrt`). Falls back to "
        "eager PyTorch fp32 if unavailable.",
    )
    parser.add_argument(
        "--tensorrt_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="TensorRT precision when --tensorrt is set (default: fp32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for the model. Defaults to $GRASPGENX_DEVICE, else cuda "
        "if available, else cpu. On cpu the ptv3vanilla backbone is forced "
        "onto its fp32 attention path (the checkpoints ship enable_flash=true, "
        "which casts q/k/v to fp16 — a CUDA-oriented path).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch intra-op threads (CPU only). Defaults to os.cpu_count().",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Must be set before graspgenx.grasp_server resolves the device.
    if args.device:
        os.environ["GRASPGENX_DEVICE"] = args.device

    import torch

    from graspgenx.grasp_server import resolve_device

    device = resolve_device()
    if device.type == "cpu":
        torch.set_num_threads(args.threads or os.cpu_count() or 1)
    logging.getLogger(__name__).info(
        "GraspGenX server device=%s threads=%d", device, torch.get_num_threads()
    )

    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    server = GraspGenXZMQServer(
        config_path=args.config,
        assets_dir=args.assets_dir,
        host=args.host,
        port=args.port,
        default_gripper=args.default_gripper,
        use_tensorrt=args.tensorrt,
        tensorrt_precision=args.tensorrt_precision,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
