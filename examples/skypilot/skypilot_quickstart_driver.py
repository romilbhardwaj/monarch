#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Running Monarch on Kubernetes / cloud VMs with SkyPilot JobGroups
=================================================================

This script demonstrates running Monarch actors on cloud infrastructure
provisioned by SkyPilot using the JobGroup abstraction. Each Monarch mesh
maps to a separate sky.Task with its own resource specification, all launched
as a managed job (sky.jobs.launch) for spot recovery and SAME_INFRA placement.

This driver must run inside the cluster (K8s pod or cloud VM). Launch it with:

    sky jobs launch monarch_driver.sky.yaml

Prerequisites:
    pip install torchmonarch-nightly
    pip install skypilot[kubernetes]  # or skypilot[aws], etc.
    sky check

Usage (from inside the cluster, or for local testing):
    # Basic: 2-node homogeneous mesh on Kubernetes
    python skypilot_quickstart_driver.py --cloud kubernetes --num-hosts 2

    # Heterogeneous: GPU trainers + CPU dataloaders
    python skypilot_quickstart_driver.py --cloud kubernetes \\
        --num-hosts 2 --accelerator H100:8 --num-dataloader-hosts 1

    # Cloud VMs
    python skypilot_quickstart_driver.py --cloud aws \\
        --num-hosts 2 --accelerator H100:1
"""

import argparse
import os
import sys

# Set timeouts before importing monarch
os.environ["HYPERACTOR_HOST_SPAWN_READY_TIMEOUT"] = "300s"
os.environ["HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT"] = "300s"
os.environ["HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE"] = "300s"

# If running inside a SkyPilot cluster, unset the in-cluster context
# to allow launching new clusters on the same Kubernetes cluster
if "SKYPILOT_IN_CLUSTER_CONTEXT_NAME" in os.environ:
    del os.environ["SKYPILOT_IN_CLUSTER_CONTEXT_NAME"]

try:
    import sky
except ImportError:
    print("ERROR: SkyPilot is not installed. Run: pip install skypilot[kubernetes]")
    sys.exit(1)

try:
    from monarch.actor import Actor, context, endpoint, ProcMesh
except ImportError as e:
    print(f"ERROR: Monarch is not properly installed: {e}")
    print("Run: pip install torchmonarch-nightly")
    sys.exit(1)

from monarch_skypilot import SkyPilotJobGroup


# ============================================================================
# Actor definitions
# ============================================================================


class Counter(Actor):
    """A simple counter actor demonstrating basic Monarch messaging."""

    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    @endpoint
    def increment(self) -> None:
        self.value += 1

    @endpoint
    def get_value(self) -> int:
        return self.value


class Trainer(Actor):
    """A trainer actor demonstrating distributed training patterns."""

    @endpoint
    def step(self) -> str:
        my_point = context().message_rank
        return f"Trainer {my_point} taking a step."

    @endpoint
    def get_info(self) -> str:
        rank = context().actor_instance.rank
        return f"Trainer at rank {rank}"


# ============================================================================
# Cloud helpers
# ============================================================================


def get_cloud(cloud_name: str):
    clouds = {
        "kubernetes": sky.Kubernetes,
        "aws": sky.AWS,
        "gcp": sky.GCP,
        "azure": sky.Azure,
        "nebius": sky.Nebius,
    }
    name = cloud_name.lower()
    if name not in clouds:
        raise ValueError(f"Unknown cloud: {cloud_name}. Available: {list(clouds.keys())}")
    return clouds[name]()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Monarch Getting Started with SkyPilot JobGroups"
    )
    parser.add_argument("--cloud", default="kubernetes",
                        help="Cloud provider (kubernetes, aws, gcp, azure, ...)")
    parser.add_argument("--num-hosts", type=int, default=2,
                        help="Number of trainer nodes")
    parser.add_argument("--gpus-per-host", type=int, default=1,
                        help="GPU processes per trainer node")
    parser.add_argument("--accelerator", default="H200:1",
                        help="GPU accelerator spec (e.g. H100:1, A100:1)")
    parser.add_argument("--num-dataloader-hosts", type=int, default=0,
                        help="Number of CPU dataloader nodes (0 = no dataloader mesh)")
    parser.add_argument("--job-name", default="monarch-getting-started",
                        help="Name for the managed job group")
    parser.add_argument("--region", default=None,
                        help="Cloud region or Kubernetes context")
    args = parser.parse_args()

    cpu_only = args.gpus_per_host == 0 or args.accelerator.lower() == "none"

    print("=" * 60)
    print("Monarch Getting Started with SkyPilot JobGroups")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Cloud:      {args.cloud}")
    print(f"  Trainers:   {args.num_hosts} nodes", end="")
    if not cpu_only:
        print(f" x {args.accelerator}")
    else:
        print(" (CPU only)")
    if args.num_dataloader_hosts > 0:
        print(f"  Dataloaders: {args.num_dataloader_hosts} CPU nodes")
    print(f"  Job name:   {args.job_name}")
    if args.region:
        print(f"  Region:     {args.region}")

    # ------------------------------------------------------------------
    # Build resource specs
    # ------------------------------------------------------------------
    trainer_resources_kwargs = {"cloud": get_cloud(args.cloud)}
    if not cpu_only:
        trainer_resources_kwargs["accelerators"] = args.accelerator
    if args.region:
        trainer_resources_kwargs["region"] = args.region
    trainer_resources = sky.Resources(**trainer_resources_kwargs)

    # Build meshes dict — heterogeneous if dataloaders are requested
    if args.num_dataloader_hosts > 0:
        dataloader_resources = sky.Resources(
            cloud=get_cloud(args.cloud),
            **({"region": args.region} if args.region else {}),
        )
        meshes = {
            "trainers":    (args.num_hosts, trainer_resources),
            "dataloaders": (args.num_dataloader_hosts, dataloader_resources),
        }
        primary_meshes = ["trainers"]
        termination_delays = {"dataloaders": "30s"}
    else:
        meshes = {"trainers": args.num_hosts}
        primary_meshes = None
        termination_delays = None

    # ------------------------------------------------------------------
    # Create job group
    # ------------------------------------------------------------------
    print("\n[1] Creating SkyPilotJobGroup...")
    job = SkyPilotJobGroup(
        meshes=meshes,
        default_resources=trainer_resources,  # used for plain-int meshes
        job_name=args.job_name,
        primary_meshes=primary_meshes,
        termination_delays=termination_delays,
    )

    try:
        # Launches all mesh tasks in parallel as a managed job group,
        # then waits for all to reach RUNNING status
        print("\n[2] Launching job group and starting Monarch workers...")
        state = job.state()

        hosts = state.trainers
        print(f"    Trainer host mesh extent: {hosts.extent}")

        # ------------------------------------------------------------------
        # Spawn processes and actors
        # ------------------------------------------------------------------
        print("\n[3] Spawning processes on trainer hosts...")
        if cpu_only:
            procs: ProcMesh = hosts.spawn_procs(per_host={"procs": 1})
        else:
            procs: ProcMesh = hosts.spawn_procs(per_host={"gpus": args.gpus_per_host})
        print(f"    Process mesh extent: {procs.extent}")

        print("\n[4] Spawning Counter actors...")
        counters: Counter = procs.spawn("counters", Counter, initial_value=0)

        print("\n[5] Broadcasting increment to all counters...")
        counters.increment.broadcast()
        counters.increment.broadcast()
        counters.increment.broadcast()

        print("\n[6] Getting counter values...")
        values = counters.get_value.call().get()
        print(f"    Counter values: {values}")
        # ValueMesh iterates as (rank_dict, value) tuples
        flat_values = [v for _, v in values]
        assert all(v == 3 for v in flat_values), f"Expected all 3, got {flat_values}"
        print("    OK — all counters are 3")

        print("\n[7] Spawning Trainer actors...")
        trainers: Trainer = procs.spawn("trainers", Trainer)

        print("\n[8] Performing distributed training step...")
        results = trainers.step.call().get()
        for r in results:
            print(f"    {r}")

        print("\n[9] Getting trainer info...")
        info = trainers.get_info.call().get()
        for i in info:
            print(f"    {i}")

        print("\n" + "=" * 60)
        print("Success! Monarch actors ran on SkyPilot JobGroup cluster!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n[10] ERROR — not cleaning up job for debugging.")
        print(f"    To view logs:  sky jobs logs {args.job_name}")
        print(f"    To clean up:   sky jobs cancel {args.job_name}")
        raise
    else:
        print("\n[10] Cancelling managed job group...")
        job.kill()
        print("    Job group cancelled.")


if __name__ == "__main__":
    main()
