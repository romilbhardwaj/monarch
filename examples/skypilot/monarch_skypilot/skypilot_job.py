# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Monarch JobTrait implementation for SkyPilot using JobGroups.

SkyPilotJobGroup allows running Monarch on Kubernetes and cloud VMs via SkyPilot,
using SkyPilot's JobGroup abstraction (sky.Dag with parallel execution) to map
each Monarch mesh to a separate sky.Task with its own resource specification.

Requirements:
    - pip install torchmonarch-nightly (or torchmonarch)
    - pip install skypilot[kubernetes] (or other cloud backends)
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from monarch._src.job.job import JobState, JobTrait

# If running inside a SkyPilot cluster, unset the in-cluster context variable
# to allow launching new clusters on the same Kubernetes cluster.
# This must be done before importing sky to affect the API server.
if "SKYPILOT_IN_CLUSTER_CONTEXT_NAME" in os.environ:
    del os.environ["SKYPILOT_IN_CLUSTER_CONTEXT_NAME"]

if TYPE_CHECKING:
    import sky
    from sky.schemas.api import responses as sky_responses

try:
    import sky
    import sky.jobs as sky_jobs
    from sky.dag import DagExecution as _DagExecution

    HAS_SKYPILOT = True
except ImportError:
    HAS_SKYPILOT = False
    sky = None  # type: ignore[assignment]
    sky_jobs = None  # type: ignore[assignment]
    _DagExecution = None  # type: ignore[assignment]


logger: logging.Logger = logging.getLogger(__name__)

# Default port for Monarch TCP communication
MONARCH_WORKER_PORT = 22222

# Timeout for waiting for all job group tasks to reach RUNNING status.
JOB_TIMEOUT = 600  # seconds

# Default setup commands to install Monarch from PyPI on remote workers.
# Requires a Docker image with Ubuntu 22.04+ with RDMA dependencies.
# For faster cold starts (<30s), use a custom Docker image with Monarch pre-installed.
DEFAULT_SETUP_COMMANDS = """
set -ex

# Install torchmonarch from PyPI
uv pip install --system torchmonarch-nightly

echo "Done installing Monarch"
"""
DEFAULT_IMAGE_ID = "docker:pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"

# Type alias: mesh spec is either just a node count or (count, Resources)
MeshSpec = Union[int, Tuple[int, "sky.Resources"]]


def _configure_transport() -> None:
    """Configure the Monarch transport using the public API."""
    from monarch.actor import enable_transport

    enable_transport("tcp")


def _attach_to_workers_wrapper(name: str, ca: str, workers: List[str]):
    """Wrapper around attach_to_workers with deferred import."""
    from monarch._src.actor.bootstrap import attach_to_workers

    return attach_to_workers(name=name, ca=ca, workers=workers)


def _resolve_meshes(
    meshes: Dict[str, MeshSpec],
    default_resources: Optional["sky.Resources"],
) -> Dict[str, Tuple[int, "sky.Resources"]]:
    """Normalize meshes dict to Dict[name, (num_nodes, Resources)]."""
    resolved: Dict[str, Tuple[int, "sky.Resources"]] = {}
    fallback = default_resources or sky.Resources(image_id=DEFAULT_IMAGE_ID)
    for name, spec in meshes.items():
        if isinstance(spec, int):
            resources = fallback
            num_nodes = spec
        else:
            num_nodes, resources = spec
        # Ensure DEFAULT_IMAGE_ID is set if no image specified
        if resources.image_id is None:
            resources = resources.copy(image_id=DEFAULT_IMAGE_ID)
        resolved[name] = (num_nodes, resources)
    return resolved


class SkyPilotJobGroup(JobTrait):
    """
    SkyPilotJobGroup provisions and manages Monarch workers on K8s and cloud VMs
    using SkyPilot's JobGroup abstraction.

    Each named Monarch mesh maps to a sky.Task in a sky.Dag(execution=PARALLEL).
    This enables:
      - Heterogeneous resources: each mesh can have its own sky.Resources
      - SAME_INFRA placement: all meshes land on the same cloud + region
      - Spot recovery: sky.jobs.launch() automatically recovers from preemptions
      - Lifecycle management: primary_meshes / termination_delays control shutdown

    Worker addresses are discovered via sky.jobs.queue_v2() (not from cluster
    handles), using internal_services K8s DNS (stable across preemptions) or
    internal_external_ips as a fallback.

    The driver that instantiates this class must itself run inside the cluster
    (K8s pod or cloud VM). When using a remote SkyPilot API server, set
    api_access: true on the driver task YAML; when using a local API server,
    omit it (SkyPilot will auto-start a local server inside the driver pod).

    Example (homogeneous):
        >>> import sky
        >>> from monarch_skypilot import SkyPilotJobGroup
        >>>
        >>> job = SkyPilotJobGroup(
        ...     meshes={"trainers": 4},
        ...     default_resources=sky.Resources(
        ...         cloud=sky.Kubernetes(), accelerators="H100:8"
        ...     ),
        ... )
        >>> state = job.state()
        >>> trainers = state.trainers  # HostMesh with 4 nodes

    Example (heterogeneous):
        >>> job = SkyPilotJobGroup(
        ...     meshes={
        ...         "trainers":    (4, sky.Resources(accelerators="H100:8")),
        ...         "dataloaders": (2, sky.Resources(cpus="32")),
        ...     },
        ...     primary_meshes=["trainers"],
        ...     termination_delays={"dataloaders": "30s"},
        ... )
    """

    def __init__(
        self,
        meshes: Dict[str, MeshSpec],
        default_resources: Optional["sky.Resources"] = None,
        job_name: Optional[str] = None,
        primary_meshes: Optional[List[str]] = None,
        termination_delays: Optional[Dict[str, str]] = None,
        monarch_port: int = MONARCH_WORKER_PORT,
        python_exe: str = "python",
        setup_commands: Optional[str] = None,
        workdir: Optional[str] = None,
        file_mounts: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            meshes: Dict mapping mesh names to either:
                - int: number of nodes (uses default_resources)
                - (int, sky.Resources): number of nodes + per-mesh resources
              e.g. {"trainers": 4}
              e.g. {"trainers": (4, sky.Resources(accelerators="H100:8")),
                    "dataloaders": (2, sky.Resources(cpus="32"))}
            default_resources: Fallback sky.Resources for meshes specified as
                               plain int. Required if any mesh uses int form.
            job_name: Name for the managed job group. Auto-generated if None.
            primary_meshes: Names of "primary" meshes. When all primary tasks
                            complete, auxiliary meshes are terminated. If None,
                            all meshes are primary.
            termination_delays: Grace period before auxiliary meshes are killed
                                after primary meshes complete.
                                e.g. {"dataloaders": "30s"} or {"default": "1m"}
            monarch_port: TCP port for Monarch worker communication.
            python_exe: Python executable on remote nodes.
            setup_commands: Setup script run before workers start. Defaults to
                           installing torchmonarch-nightly from PyPI.
            workdir: Local directory to sync to ~/sky_workdir on each node.
            file_mounts: Additional file mounts {remote_path: local_path}.
        """
        if not HAS_SKYPILOT:
            raise ImportError(
                "SkyPilot is not installed. Install with: pip install skypilot[kubernetes]"
            )

        try:
            _configure_transport()
        except ImportError:
            pass

        super().__init__()

        self._meshes_input = meshes
        self._default_resources = default_resources
        self._job_name = job_name or f"monarch-{os.getpid()}"
        self._primary_meshes = primary_meshes
        self._termination_delays = termination_delays
        self._port = monarch_port
        self._python_exe = python_exe
        self._setup_commands = setup_commands
        self._workdir = workdir
        self._file_mounts = file_mounts

        # Resolved at _create() time
        self._resolved: Optional[Dict[str, Tuple[int, "sky.Resources"]]] = None

    # ------------------------------------------------------------------
    # JobTrait implementation
    # ------------------------------------------------------------------

    def _create(self, client_script: Optional[str]) -> None:
        """Build a JobGroup dag and launch it as a managed job."""
        if client_script is not None:
            raise RuntimeError("SkyPilotJobGroup cannot run batch-mode scripts yet")

        self._resolved = _resolve_meshes(self._meshes_input, self._default_resources)

        worker_cmd = self._build_worker_command()
        setup = self._setup_commands if self._setup_commands is not None else DEFAULT_SETUP_COMMANDS
        if setup and not setup.endswith("\n"):
            setup += "\n"

        dag = sky.Dag()
        dag.name = self._job_name
        dag.set_execution(_DagExecution.PARALLEL)

        for mesh_name, (num_nodes, resources) in self._resolved.items():
            task = sky.Task(
                name=mesh_name,
                setup=setup or None,
                run=worker_cmd,
                num_nodes=num_nodes,
                workdir=self._workdir,
            )
            if self._file_mounts:
                task.set_file_mounts(self._file_mounts)
            task.set_resources(resources)
            dag.add(task)

        if self._primary_meshes:
            dag.primary_tasks = self._primary_meshes
        if self._termination_delays:
            dag.termination_delay = self._termination_delays

        logger.info(
            f"Launching JobGroup '{self._job_name}' with meshes: "
            + ", ".join(f"{n}={c}" for n, (c, _) in self._resolved.items())
        )

        try:
            request_id = sky_jobs.launch(dag, name=self._job_name)
            sky.get(request_id)
        except Exception as e:
            logger.error(f"Failed to launch JobGroup '{self._job_name}': {e}")
            raise RuntimeError(f"Failed to launch JobGroup: {e}") from e

        logger.info(f"JobGroup '{self._job_name}' submitted, waiting for workers...")
        self._wait_for_all_tasks_running(timeout=JOB_TIMEOUT)

    def _state(self) -> JobState:
        """Connect to workers and return JobState with HostMesh per mesh."""
        if self._resolved is None:
            raise RuntimeError("JobGroup has not been created yet")

        host_meshes = {}
        for mesh_name, (num_nodes, _) in self._resolved.items():
            workers = self._get_worker_addrs(mesh_name, num_nodes)
            logger.info(f"Connecting to mesh '{mesh_name}' workers: {workers}")

            host_mesh = _attach_to_workers_wrapper(
                name=mesh_name,
                ca="trust_all_connections",
                workers=workers,
            )
            logger.info(f"Waiting for mesh '{mesh_name}' to initialize...")
            host_mesh.initialized.get()
            logger.info(f"Mesh '{mesh_name}' ready")

            host_meshes[mesh_name] = host_mesh

        return JobState(host_meshes)

    def _kill(self) -> None:
        """Cancel the managed job group."""
        logger.info(f"Cancelling JobGroup '{self._job_name}'")
        try:
            request_id = sky_jobs.cancel(name=self._job_name)
            sky.get(request_id)
            logger.info(f"JobGroup '{self._job_name}' cancelled")
        except Exception as e:
            logger.warning(f"Failed to cancel JobGroup '{self._job_name}': {e}")

    def can_run(self, spec: "JobTrait") -> bool:
        """Check if this running job matches the given spec."""
        if not isinstance(spec, SkyPilotJobGroup):
            return False
        if not self.active:
            return False
        return (
            spec._meshes_input == self._meshes_input
            and spec._default_resources == self._default_resources
            and spec._port == self._port
            and self._is_job_running()
        )

    # ------------------------------------------------------------------
    # Worker discovery (via queue_v2, PR #8735)
    # ------------------------------------------------------------------

    def _get_worker_addrs(self, mesh_name: str, num_nodes: int) -> List[str]:
        """Get TCP worker addresses for a mesh using queue_v2.

        On K8s uses internal_services DNS (stable across spot preemptions since
        the K8s service reroutes to the new pod). Falls back to
        internal_external_ips on all clouds.

        Note on task_name: For true multi-task JobGroups, each task record has
        task_name = the Dag task name (e.g. "trainers"). For single-task managed
        jobs (or single-task JobGroups), SkyPilot sets task_name = job_name.
        We handle both cases.
        """
        from sky.jobs.client import sdk as jobs_sdk

        request_id = jobs_sdk.queue_v2(refresh=False, skip_finished=False)
        records, _, _, _ = sky.get(request_id)

        all_named = [r for r in records if r.job_name == self._job_name]

        # When a job name is reused, multiple records exist. Use only the latest.
        if all_named:
            latest_job_id = max(r.job_id for r in all_named)
            job_records = [r for r in all_named if r.job_id == latest_job_id]
        else:
            job_records = []

        # Multi-task JobGroup: task_name matches the Dag task/mesh name
        record = next((r for r in job_records if r.task_name == mesh_name), None)

        # Single-task fallback: SkyPilot uses job_name as task_name when there
        # is only one task. This happens when the JobGroup has a single mesh.
        if record is None and len(self._resolved) == 1:  # type: ignore[arg-type]
            record = next(
                (r for r in job_records if r.task_name == self._job_name), None
            )

        if record is None:
            raise RuntimeError(
                f"No job record found for mesh '{mesh_name}' in job '{self._job_name}'. "
                f"Is the job still running?"
            )

        # K8s: internal_services gives stable *.svc.cluster.local DNS names,
        # resolvable cluster-wide (not just from within the job group).
        # (Added in SkyPilot PR #8735; use getattr for compat with older versions.)
        internal_services = getattr(record, "internal_services", None)
        if internal_services and len(internal_services) >= num_nodes:
            return [
                f"tcp://{dns}:{self._port}"
                for dns in internal_services.values()
                if dns
            ]

        # Fallback: raw IPs from internal_external_ips (all clouds).
        # Prefer external IP; fall back to internal.
        # (Also added in PR #8735; use getattr for compat.)
        internal_external_ips = getattr(record, "internal_external_ips", None)
        if internal_external_ips:
            return [
                f"tcp://{ext or intern_}:{self._port}"
                for intern_, ext in internal_external_ips
            ]

        raise RuntimeError(
            f"No IP or DNS information available for mesh '{mesh_name}'. "
            f"The job may still be starting."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_worker_command(self) -> str:
        """Build the bash command to start a Monarch worker on each node."""
        python_code = f"""
import socket
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)
address = f"tcp://{{ip_addr}}:{self._port}"
print(f"Starting Monarch worker at {{address}} (hostname={{hostname}})", flush=True)

try:
    from monarch.actor import run_worker_loop_forever
    print("Worker ready and listening...", flush=True)
    run_worker_loop_forever(address=address, ca="trust_all_connections")
except Exception as e:
    print(f"ERROR in worker: {{e}}", flush=True)
    import traceback
    traceback.print_exc()
    raise
"""
        escaped = python_code.replace("'", "'\"'\"'")
        env_vars = (
            f"export HYPERACTOR_HOST_SPAWN_READY_TIMEOUT={JOB_TIMEOUT}s && "
            f"export HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT={JOB_TIMEOUT}s && "
            f"export HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE={JOB_TIMEOUT}s"
        )
        return f"{env_vars} && {self._python_exe} -c '{escaped}'"

    def _wait_for_all_tasks_running(self, timeout: int = JOB_TIMEOUT) -> None:
        """Poll queue_v2 until all mesh tasks are RUNNING."""
        from sky.jobs.client import sdk as jobs_sdk
        from sky.jobs import state as managed_job_state

        expected = set(self._resolved.keys())  # type: ignore[union-attr]
        start = time.time()
        poll_interval = 10

        while time.time() - start < timeout:
            try:
                request_id = jobs_sdk.queue_v2(refresh=False, skip_finished=False)
                records, _, _, _ = sky.get(request_id)

                all_named = [r for r in records if r.job_name == self._job_name]

                if not all_named:
                    elapsed = int(time.time() - start)
                    logger.info(f"Waiting for job '{self._job_name}' to appear ({elapsed}s)...")
                    time.sleep(poll_interval)
                    continue

                # When a job name is reused, multiple records exist. Use only
                # the latest (highest job_id) to avoid matching stale runs.
                latest_job_id = max(r.job_id for r in all_named)
                job_records = [r for r in all_named if r.job_id == latest_job_id]

                # Check for terminal failures
                for r in job_records:
                    if r.status in (
                        managed_job_state.ManagedJobStatus.FAILED,
                        managed_job_state.ManagedJobStatus.FAILED_SETUP,
                        managed_job_state.ManagedJobStatus.CANCELLED,
                    ):
                        raise RuntimeError(
                            f"Task '{r.task_name}' in job '{self._job_name}' "
                            f"failed with status: {r.status}. "
                            f"Check logs with: sky jobs logs {self._job_name}"
                        )

                running_task_names = {
                    r.task_name
                    for r in job_records
                    if r.status == managed_job_state.ManagedJobStatus.RUNNING
                }

                # Map running task names to mesh names.
                # Multi-task: task_name == mesh_name.
                # Single-task: task_name == job_name; map to the only mesh.
                running = set()
                for mesh in expected:
                    if mesh in running_task_names:
                        running.add(mesh)
                    elif (len(expected) == 1
                          and self._job_name in running_task_names):
                        running.add(mesh)

                if expected <= running:
                    logger.info(
                        f"All tasks in '{self._job_name}' are RUNNING: {sorted(running)}"
                    )
                    return

                elapsed = int(time.time() - start)
                waiting_for = expected - running
                logger.info(
                    f"Waiting for tasks {waiting_for} to reach RUNNING ({elapsed}s)..."
                )

            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(f"Error polling job status: {e}")

            time.sleep(poll_interval)

        raise RuntimeError(
            f"Timeout after {timeout}s waiting for all tasks in '{self._job_name}' "
            f"to reach RUNNING status."
        )

    def _is_job_running(self) -> bool:
        """Return True if the job group is still running."""
        from sky.jobs.client import sdk as jobs_sdk
        from sky.jobs import state as managed_job_state

        try:
            request_id = jobs_sdk.queue_v2(refresh=False, skip_finished=False)
            records, _, _, _ = sky.get(request_id)
            job_records = [r for r in records if r.job_name == self._job_name]
            return any(
                r.status == managed_job_state.ManagedJobStatus.RUNNING
                for r in job_records
            )
        except Exception as e:
            logger.warning(f"Error checking job status: {e}")
            return False
