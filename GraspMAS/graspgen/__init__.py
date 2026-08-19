"""Client-side bridge from GraspMAS to a GraspGen-X inference server.

Deliberately torch-free and asset-free: this package only speaks the msgpack /
ZMQ wire protocol, so it can live in the GraspMAS environment without dragging
in GraspGen-X's pinned `diffusers` / `huggingface-hub` stack.
"""

from graspgen.client import GraspGenClient, GraspGenUnavailable

__all__ = ["GraspGenClient", "GraspGenUnavailable"]
