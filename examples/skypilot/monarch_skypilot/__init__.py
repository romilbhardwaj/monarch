# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monarch SkyPilot Integration Package.

This package provides SkyPilotJobGroup - a way to run Monarch workloads on
Kubernetes and cloud VMs via SkyPilot's JobGroup abstraction.

Each named Monarch mesh maps to a sky.Task with its own resource specification,
all launched as a managed job group (sky.jobs.launch) for spot recovery and
SAME_INFRA placement.

Usage:
    from monarch_skypilot import SkyPilotJobGroup

    # Homogeneous: all meshes share the same resources
    job = SkyPilotJobGroup(
        meshes={"trainers": 4},
        default_resources=sky.Resources(cloud=sky.Kubernetes(), accelerators="H100:8"),
    )

    # Heterogeneous: per-mesh resource specifications
    job = SkyPilotJobGroup(
        meshes={
            "trainers":    (4, sky.Resources(accelerators="H100:8")),
            "dataloaders": (2, sky.Resources(cpus="32")),
        },
        primary_meshes=["trainers"],
        termination_delays={"dataloaders": "30s"},
    )

    state = job.state()
    trainers = state.trainers  # HostMesh with 4 nodes
"""

from .skypilot_job import SkyPilotJobGroup

__all__ = ["SkyPilotJobGroup"]
