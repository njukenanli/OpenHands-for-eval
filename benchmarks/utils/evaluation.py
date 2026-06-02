"""
Evaluation orchestrator.

This module provides async-based evaluation orchestration for benchmarks.
The evaluation uses asyncio for concurrent instance processing, running
synchronous SDK operations in thread executors. This eliminates the 30×
memory multiplication from ProcessPoolExecutor while maintaining high
concurrency for I/O-bound workloads (HTTP calls to LLM proxy + runtime API).
"""

import asyncio
import base64
import io
import json
import os
import shutil
import tarfile
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional, Tuple
from uuid import UUID

from lmnr import Laminar
from pydantic import BaseModel, Field
from tqdm import tqdm

from benchmarks.utils.acp import is_acp_agent
from benchmarks.utils.failure_classifier import FailureCategory, classify_failure
from benchmarks.utils.laminar import LMNR_ENV_VARS, LaminarEvalMetadata, LaminarService
from benchmarks.utils.litellm_proxy import (
    create_virtual_key,
    delete_key,
    get_key_spend,
    set_current_virtual_key,
)
from benchmarks.utils.models import (
    EvalInstance,
    EvalInstanceID,
    EvalMetadata,
    EvalOutput,
    RemoteRuntimeAllocation,
)
from openhands.sdk import (
    ConversationStats,
    __version__ as openhands_sdk_version,
    get_logger,
)
from openhands.sdk.critic import CriticBase
from openhands.sdk.llm import Metrics
from openhands.sdk.workspace import RemoteWorkspace
from openhands.workspace import APIRemoteWorkspace


logger = get_logger(__name__)

# Interval in seconds between checking for per-instance timeouts
TIMEOUT_CHECK_INTERVAL_SECONDS = 60


def _default_thread_pool_workers(cpu_count: int | None) -> int:
    """Return Python's implicit ThreadPoolExecutor size for the given CPU count."""
    return min(32, (cpu_count or 1) + 4)


def _read_cgroup_cpu_max() -> str | None:
    """Return the cgroup v2 cpu.max value when available."""
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    try:
        if cpu_max_path.exists():
            return cpu_max_path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return None


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types.

    Pandas ``row.to_dict()`` preserves numpy dtypes, which Pydantic's
    ``model_dump_json`` cannot serialise.  This is used by
    ``_create_error_output`` to sanitise ``instance.data`` before storing it
    in an ``EvalOutput``.
    """
    try:
        import numpy as np
    except ImportError:
        return obj

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_serializable(v) for v in obj)
    return obj


@dataclass
class PendingInstance:
    """Tracks state for a pending evaluation instance."""

    instance: EvalInstance
    datapoint_id: UUID | None = None
    task: asyncio.Task | None = field(default=None, repr=False)
    start_time: float | None = None  # Set when worker thread begins executing


OnResult = Callable[[EvalInstance, EvalOutput], Coroutine[Any, Any, None]]


class Evaluation(ABC, BaseModel):
    """Abstract orchestrator for instance processing using asyncio.

    Uses asyncio for concurrent instance processing with a semaphore to limit
    the number of concurrent instances. Synchronous SDK operations (workspace,
    conversation) are run in thread executors via asyncio.to_thread().

    This design eliminates the memory multiplication from ProcessPoolExecutor
    while maintaining high concurrency for I/O-bound workloads.
    """

    metadata: EvalMetadata
    num_workers: int = Field(default=1, ge=1)
    max_asyncio_thread_workers: int = Field(
        default=20,
        ge=1,
        description=(
            "Upper bound for the asyncio default thread pool used by "
            "asyncio.to_thread(). This prevents accidental misconfiguration "
            "from creating an unbounded number of threads."
        ),
    )
    current_attempt: int = Field(
        default=1, description="Current attempt number (1-indexed)"
    )
    instance_timeout: int = Field(
        default=4 * 60 * 60,  # 4 hours
        description=(
            "Maximum time in seconds for a single instance to complete. "
            "When a timeout occurs, the instance's asyncio task is cancelled. "
            "The underlying thread running the SDK operation will complete, "
            "but the result will be discarded and replaced with a timeout error."
        ),
    )

    def model_post_init(self, __context) -> None:
        """Stamp openhands_sdk_version on self.metadata and persist metadata.json."""
        if self.metadata.n_critic_runs != 1:
            logger.warning(
                "Forcing n_critic_runs from %d to 1; retries are disabled",
                self.metadata.n_critic_runs,
            )
            self.metadata.n_critic_runs = 1
        if self.metadata.max_retries != 0:
            logger.warning(
                "Forcing max_retries from %d to 0; retries are disabled",
                self.metadata.max_retries,
            )
            self.metadata.max_retries = 0
        self.metadata.openhands_sdk_version = openhands_sdk_version
        self._save_metadata()

    def _save_metadata(self) -> None:
        os.makedirs(self.metadata.eval_output_dir, exist_ok=True)
        metadata_file = os.path.join(self.metadata.eval_output_dir, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(self.metadata.model_dump_json(indent=2))
        logger.info(f"Saved metadata to {metadata_file}")

    def _stamp_acp_metadata_from_outputs(self, outputs: List[EvalOutput]) -> None:
        """Back-write ACP handshake fields from any completed instance."""
        if not is_acp_agent(self.metadata.agent_type):
            return
        for out in outputs:
            name = out.test_result.get("acp_agent_name")
            version = out.test_result.get("acp_agent_version")
            if name and version:
                self.metadata.acp_agent_name = name
                self.metadata.acp_agent_version = version
                self._save_metadata()
                logger.info("Stamped ACP metadata: name=%r version=%r", name, version)
                return
        completed = sum(1 for out in outputs if out.test_result)
        logger.warning(
            "ACP run: %d/%d instances completed but none surfaced "
            "acp_agent_name+acp_agent_version. push-to-index will see a "
            "missing agent_version.",
            completed,
            len(outputs),
        )

    def _conversation_dir(self, instance_id: str) -> Path:
        return Path(self.metadata.eval_output_dir) / "conversations" / instance_id

    def _trajectory_exists(self, instance_id: str) -> bool:
        trajectory_path = self._conversation_dir(instance_id) / "trajectory.json"
        return trajectory_path.is_file() and trajectory_path.stat().st_size > 0

    def _patch_path(self, instance_id: str) -> Path:
        return self._conversation_dir(instance_id) / "patch.diff"

    def _read_patch(self, instance_id: str) -> str:
        patch_path = self._patch_path(instance_id)
        if not patch_path.is_file() or patch_path.stat().st_size == 0:
            return ""
        return patch_path.read_text(encoding="utf-8")

    def _write_patch_file(self, out: EvalOutput) -> Path:
        patch = (out.test_result or {}).get("git_patch") or ""
        conversation_dir = self._conversation_dir(out.instance_id)
        conversation_dir.mkdir(parents=True, exist_ok=True)
        patch_path = conversation_dir / "patch.diff"
        patch_path.write_text(patch, encoding="utf-8")
        return patch_path

    def _write_predictions_from_patch_files(self) -> Path:
        predictions: dict[str, dict[str, str]] = {}
        conversations_dir = Path(self.metadata.eval_output_dir) / "conversations"
        if conversations_dir.is_dir():
            for patch_path in sorted(conversations_dir.glob("*/patch.diff")):
                instance_id = patch_path.parent.name
                patch = self._read_patch(instance_id)
                if not patch.strip() or not self._trajectory_exists(instance_id):
                    continue
                predictions[instance_id] = {
                    "instance_id": instance_id,
                    "model_patch": patch,
                }

        preds_path = Path(self.metadata.eval_output_dir) / "preds.json"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with preds_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        return preds_path

    def _instances_with_non_empty_patch_and_trajectory(self) -> set[EvalInstanceID]:
        completed_instances: set[EvalInstanceID] = set()
        conversations_dir = Path(self.metadata.eval_output_dir) / "conversations"
        if conversations_dir.is_dir():
            for patch_path in sorted(conversations_dir.glob("*/patch.diff")):
                instance_id = patch_path.parent.name
                if self._read_patch(instance_id).strip() and self._trajectory_exists(instance_id):
                    completed_instances.add(instance_id)
        logger.info(
            "Found %d restart-complete instances in %s",
            len(completed_instances),
            conversations_dir,
        )
        return completed_instances

    def _get_completed_instances(self) -> set[EvalInstanceID]:
        """Return instances safe to skip on restart."""
        return self._instances_with_non_empty_patch_and_trajectory()

    @abstractmethod
    def prepare_instances(self) -> List[EvalInstance]:
        """Return the list of instances to evaluate."""
        raise NotImplementedError

    @abstractmethod
    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Create and return a context-managed Workspace for the given instance.

        Args:
            instance: The evaluation instance to prepare workspace for.
            resource_factor: Resource factor for runtime allocation (default: 1).
            forward_env: Environment variables to forward into the workspace.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """Run evaluation for a single instance in the provided workspace."""
        raise NotImplementedError

    def _extract_base_state_from_conversation_archive(
        self,
        conversation_archive_path: Path,
    ) -> dict[str, Any] | None:
        """Load the persisted conversation base_state.json payload."""
        if conversation_archive_path.is_dir():
            base_state_path = next(
                conversation_archive_path.rglob("base_state.json"), None
            )
            if base_state_path is None:
                return None
            return json.loads(base_state_path.read_text())

        with tarfile.open(conversation_archive_path, mode="r:gz") as conv_tar:
            base_state_file = next(
                (
                    member
                    for member in conv_tar.getmembers()
                    if member.name.endswith("base_state.json")
                ),
                None,
            )
            if base_state_file is None:
                return None

            base_state_obj = conv_tar.extractfile(base_state_file)
            if base_state_obj is None:
                return None

            return json.loads(base_state_obj.read().decode("utf-8"))

    def _safe_extract_tar(self, conv_tar: tarfile.TarFile, target_dir: Path) -> None:
        """Extract a tar into target_dir without allowing path traversal."""
        target_root = target_dir.resolve()
        for member in conv_tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if member_path != target_root and target_root not in member_path.parents:
                raise RuntimeError(f"Unsafe conversation archive member: {member.name}")
        conv_tar.extractall(target_dir, filter="data")

    def _write_trajectory_files(self, conversation_dir: Path) -> Path | None:
        """Write persisted event JSON files to a trajectory JSON array."""
        event_paths = sorted(conversation_dir.rglob("events/event-*.json"))
        if not event_paths:
            return None

        events: list[Any] = []
        for event_path in event_paths:
            events.append(json.loads(event_path.read_text()))

        json_path = conversation_dir / "trajectory.json"
        with json_path.open("w", encoding="utf-8") as out_file:
            json.dump(events, out_file, indent=2)
        return json_path

    def _has_meaningful_metrics(self, metrics: Metrics) -> bool:
        """Return whether recovered metrics contain non-zero cost or usage."""
        if metrics.accumulated_cost > 0:
            return True

        usage = metrics.accumulated_token_usage
        return usage is not None and any(
            (
                usage.prompt_tokens > 0,
                usage.completion_tokens > 0,
                usage.cache_read_tokens > 0,
                usage.cache_write_tokens > 0,
                usage.reasoning_tokens > 0,
            )
        )

    def _load_metrics_from_conversation_archive(
        self,
        conversation_archive_path: Path | None,
    ) -> Metrics | None:
        """Recover aggregated metrics from a saved conversation path."""
        if conversation_archive_path is None or not conversation_archive_path.exists():
            return None

        try:
            base_state = self._extract_base_state_from_conversation_archive(
                conversation_archive_path
            )
            if base_state is None:
                return None

            stats = ConversationStats.model_validate((base_state.get("stats") or {}))
            metrics = stats.get_combined_metrics()
            if not self._has_meaningful_metrics(metrics):
                return None
            return metrics
        except Exception as exc:
            logger.warning(
                "[worker] Failed to recover metrics from %s: %s",
                conversation_archive_path,
                exc,
            )
            return None

    def _query_proxy_cost(
        self,
        instance_id: str,
        virtual_key: str | None,
    ) -> float | None:
        """Query exact per-instance spend from the LiteLLM proxy."""
        if virtual_key is None:
            return None

        proxy_cost = get_key_spend(virtual_key)
        if proxy_cost is None or proxy_cost == 0.0:
            logger.info(
                "[worker] proxy spend not yet available for %s, retrying...",
                instance_id,
            )
            for delay in (2, 4, 8, 16):
                time.sleep(delay)
                retry_cost = get_key_spend(virtual_key)
                if retry_cost is not None and retry_cost > 0:
                    proxy_cost = retry_cost
                    break

        if proxy_cost is not None and proxy_cost == 0.0:
            logger.warning(
                "[worker] proxy cost still $0 for %s after retries — "
                "spend may not have been committed by the proxy",
                instance_id,
            )

        return proxy_cost

    def _create_error_output(
        self,
        instance: EvalInstance,
        error: Exception,
        retry_count: int,
        metrics: Metrics | None = None,
        proxy_cost: float | None = None,
    ) -> EvalOutput:
        """Create an EvalOutput object for a failed instance."""
        test_result: dict[str, Any] = {}
        if proxy_cost is not None:
            test_result["proxy_cost"] = proxy_cost

        return EvalOutput(
            instance_id=instance.id,
            test_result=test_result,
            instruction=None,
            error=(
                f"Instance failed after {retry_count} retries. Last error: {str(error)}"
            )[:200],
            history=[],
            metrics=metrics,
            instance=_to_serializable(instance.data),
        )

    def _capture_conversation_archive(
        self,
        workspace: RemoteWorkspace,
        instance: EvalInstance,
    ) -> Path | None:
        """Capture conversation trajectory from the remote runtime.

        Persists the /workspace/conversations directory from the remote runtime
        to plain per-instance files in the evaluation output directory.

        This provides a complete record of the agent's conversation history,
        which is valuable for debugging, analysis, and reproducibility.

        Args:
            workspace: The remote workspace to capture from
            instance: The evaluation instance being processed
        """
        try:
            # Tar is only used as an internal transfer format from the container.
            # The saved artifact is an extracted directory of plain JSON files.
            conv_cmd = (
                "cd / && "
                "if [ -d workspace/conversations ]; then "
                "tar -czf - workspace/conversations | base64; "
                "else echo ''; fi"
            )
            tar_cmd = workspace.execute_command(conv_cmd)

            if tar_cmd.exit_code == 0 and tar_cmd.stdout.strip():
                conversations_dir = (
                    Path(self.metadata.eval_output_dir) / "conversations"
                )
                conversations_dir.mkdir(parents=True, exist_ok=True)
                conversation_dir = conversations_dir / instance.id

                if conversation_dir.exists():
                    shutil.rmtree(conversation_dir)
                conversation_dir.mkdir(parents=True)

                tar_bytes = base64.b64decode(tar_cmd.stdout)
                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as conv_tar:
                    self._safe_extract_tar(conv_tar, conversation_dir)
                trajectory_path = self._write_trajectory_files(conversation_dir)

                logger.info(
                    "[worker] Saved conversation files for %s to %s",
                    instance.id,
                    conversation_dir,
                )
                if trajectory_path is not None:
                    logger.info(
                        "[worker] Saved trajectory file for %s to %s",
                        instance.id,
                        trajectory_path,
                    )
                return conversation_dir

            logger.debug(
                "[worker] No conversation files for %s (directory not found or empty)",
                instance.id,
            )
            return None
        except Exception as e:
            logger.warning(
                "[worker] Failed to capture conversation trajectory for %s: %s",
                instance.id,
                e,
            )
            return None

    # --- Runner ---
    def run(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """
        Run evaluation with iterative mode support.

        If n_critic_runs > 1, will retry failed instances multiple times.
        If n_critic_runs == 1, will run once without retries.

        Uses asyncio for concurrent instance processing. Synchronous SDK
        operations run in thread executors via asyncio.to_thread().
        """
        logger.info("Starting evaluation (asyncio)")
        logger.info("metadata=%s", self.metadata)
        logger.info("workers=%d", self.num_workers)
        logger.info("n_critic_runs=%d", self.metadata.n_critic_runs)

        # Use iterative mode for all cases
        return self._run_iterative_mode(on_result=on_result)

    def _get_instances_for_attempt(
        self,
        attempt: int,
        all_instances: List[EvalInstance],
        critic: CriticBase,
    ) -> List[EvalInstance]:
        """
        Determine which instances need processing for a specific attempt.

        State is derived from the on-disk attempt files rather than kept
        in memory so that a crashed process can resume where it left off.

        This method handles all resume scenarios naturally without special cases:
        - New instances: Not completed in attempt 1 yet → include them
        - Resume: Already completed in this attempt → exclude them
        - Expansion: Just more instances not in attempt 1 yet → include them

        Args:
            attempt: The attempt number (1-indexed)
            all_instances: All instances in the dataset
            critic: The critic to use for determining failures

        Returns:
            List of instances that need processing for this attempt
        """
        if attempt == 1:
            completed_instances = self._instances_with_non_empty_patch_and_trajectory()
            return [
                inst for inst in all_instances if inst.id not in completed_instances
            ]

        return []

    def _run_iterative_mode(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """Run evaluation with support for single or multiple attempts.

        Uses asyncio for concurrent instance processing. Synchronous SDK
        operations run in thread executors via asyncio.to_thread().
        """
        return asyncio.run(self._run_iterative_mode_async(on_result=on_result))

    async def _run_iterative_mode_async(
        self,
        *,
        on_result: Optional[OnResult] = None,
    ) -> List[EvalOutput]:
        """Async implementation of iterative mode evaluation."""
        # Install thread-routed logging/stdout and set up main-thread defaults
        # before spawning any workers.
        from benchmarks.utils.worker_context import initialize as init_worker_ctx

        init_worker_ctx()

        loop = asyncio.get_running_loop()
        cpu_count = os.cpu_count()
        default_executor_workers = _default_thread_pool_workers(cpu_count)
        effective_executor_workers = min(
            self.num_workers, self.max_asyncio_thread_workers
        )
        if effective_executor_workers < self.num_workers:
            logger.warning(
                "[executor] capping configured_workers=%d to executor_cap=%d",
                self.num_workers,
                self.max_asyncio_thread_workers,
            )
        loop.set_default_executor(
            ThreadPoolExecutor(
                max_workers=effective_executor_workers,
                thread_name_prefix="evaluation-worker",
            )
        )
        logger.info(
            "[executor] configured_workers=%d executor_cap=%d effective_max_workers=%d default_max_workers=%d os_cpu_count=%s cpu.max=%s",
            self.num_workers,
            self.max_asyncio_thread_workers,
            effective_executor_workers,
            default_executor_workers,
            cpu_count,
            _read_cgroup_cpu_max() or "unknown",
        )

        all_instances = self.prepare_instances()

        # Initialize Laminar
        LaminarService.get().initialize()

        # Build metadata for Laminar evaluation and traces
        run_id = os.getenv("UNIQUE_EVAL_NAME")
        laminar_meta = {
            k: v
            for k, v in [
                ("benchmark", self.metadata.dataset),
                ("model", self.metadata.llm.model),
            ]
            if v
        }

        # Create Laminar evaluation (use run_id as name if available)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_name = (
            run_id or f"{self.metadata.dataset} {self.metadata.dataset_split} {now}"
        )
        self.metadata.lmnr = LaminarEvalMetadata(
            eval_id=LaminarService.get().create_evaluation(
                name=eval_name,
                group_name=f"{self.metadata.dataset} {self.metadata.dataset_split}",
                metadata=laminar_meta or None,
            )
        )
        # Store for use in datapoint creation
        self._laminar_session_id = run_id
        self._laminar_trace_meta = laminar_meta or None

        total_instances = len(all_instances)
        logger.info("prepared %d instances for evaluation", total_instances)

        if total_instances == 0:
            logger.warning("No instances to process.")
            return []

        critic = self.metadata.critic
        all_outputs: List[EvalOutput] = []

        for attempt in range(1, self.metadata.n_critic_runs + 1):
            self.current_attempt = attempt
            logger.info(f"Starting attempt {attempt}/{self.metadata.n_critic_runs}")

            instances_to_process = self._get_instances_for_attempt(
                attempt, all_instances, critic
            )
            if attempt == 1:
                self._write_predictions_from_patch_files()

            logger.info(f"Processing {len(instances_to_process)} instances")

            if not instances_to_process:
                logger.info("No instances to process, skipping to next attempt")
                continue

            # Adjust temperature for retries (deterministic -> non-deterministic)
            original_temperature = self.metadata.llm.temperature
            if attempt > 1 and original_temperature == 0.0:
                logger.info("Adjusting temperature from 0.0 to 0.1 for retry attempt")
                self.metadata.llm.temperature = 0.1

            attempt_outputs: List[EvalOutput] = []

            async def attempt_on_result_async(
                instance: EvalInstance, out: EvalOutput
            ) -> None:
                patch_path = self._write_patch_file(out)
                if patch_path is not None:
                    logger.info(
                        "[worker] Saved patch for %s to %s",
                        instance.id,
                        patch_path,
                    )
                self._write_predictions_from_patch_files()

                if on_result:
                    await on_result(instance, out)

                attempt_outputs.append(out)

            # Run evaluation for this attempt using asyncio
            attempt_outputs = await self._run_attempt_async(
                instances_to_process,
                attempt,
                attempt_on_result_async,
            )

            # Restore original temperature
            if attempt > 1 and original_temperature == 0.0:
                self.metadata.llm.temperature = original_temperature

            logger.info(
                f"Attempt {attempt} complete: "
                f"{len(attempt_outputs)} instances processed"
            )
            all_outputs.extend(attempt_outputs)

        logger.info(
            f"Evaluation complete: {total_instances} total instances, "
            f"{self.metadata.n_critic_runs} critic runs"
        )

        self._stamp_acp_metadata_from_outputs(all_outputs)

        return all_outputs

    async def _run_attempt_async(
        self,
        instances: List[EvalInstance],
        attempt: int,
        on_result: Callable[[EvalInstance, EvalOutput], Coroutine[Any, Any, None]],
    ) -> List[EvalOutput]:
        """Run a single attempt with async concurrency.

        Uses asyncio.Semaphore to limit concurrent instances and
        asyncio.to_thread() to run sync SDK operations.

        Args:
            instances: List of instances to process
            attempt: Current attempt number
            on_result: Async callback for each completed instance

        Returns:
            List of EvalOutput for completed instances
        """
        semaphore = asyncio.Semaphore(self.num_workers)
        pending_instances: dict[asyncio.Task, PendingInstance] = {}
        attempt_outputs: List[EvalOutput] = []
        progress = tqdm(total=len(instances), desc=f"Attempt {attempt}", leave=False)

        async def process_with_semaphore(
            inst: EvalInstance,
            datapoint_id: UUID | None,
        ) -> Tuple[EvalInstance, EvalOutput]:
            """Process one instance with semaphore-based concurrency control."""
            async with semaphore:
                task = asyncio.current_task()
                pending_info = pending_instances.get(task) if task is not None else None

                def _thread_wrapper() -> Tuple[EvalInstance, EvalOutput]:
                    # Record start time when the thread actually begins
                    # executing, not when the semaphore was acquired. This
                    # avoids counting thread-pool queue time against the
                    # per-instance timeout.
                    if pending_info is not None:
                        pending_info.start_time = time.monotonic()
                    return self._process_one_sync(
                        inst,
                        attempt,
                        lmnr_session_id=self._laminar_session_id,
                        lmnr_trace_metadata=self._laminar_trace_meta,
                        lmnr_datapoint_id=datapoint_id,
                    )

                # Run the sync processing function in a thread
                return await asyncio.to_thread(_thread_wrapper)

        # Create all tasks
        tasks: list[asyncio.Task] = []
        # lmnr is guaranteed to be set in _run_iterative_mode_async before this call
        assert self.metadata.lmnr is not None
        for index, inst in enumerate(instances):
            # Two-phase datapoint linking:
            # 1. Create datapoint immediately (for UI progress tracking)
            # 2. Worker starts eval_span when work begins (accurate timeline)
            # 3. Link them via update_datapoint_trace_id when worker starts
            datapoint_id = LaminarService.get().create_evaluation_datapoint(
                self.metadata.lmnr.eval_id,
                inst.id,
                self.metadata.model_dump(mode="json"),
                index,
            )

            task = asyncio.create_task(process_with_semaphore(inst, datapoint_id))
            tasks.append(task)
            pending_instances[task] = PendingInstance(
                instance=inst,
                datapoint_id=datapoint_id,
                task=task,
            )

        # Process tasks as they complete with timeout checking
        pending: set[asyncio.Task] = set(tasks)

        while pending:
            # Wait for either a task to complete or timeout interval
            done, pending = await asyncio.wait(
                pending,
                timeout=TIMEOUT_CHECK_INTERVAL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process completed tasks
            for task in done:
                progress.update(1)
                pending_info = pending_instances.get(task)
                try:
                    instance, out = task.result()

                    # Add Laminar metadata to EvalOutput
                    if out.metadata is None:
                        out.metadata = self.metadata.model_copy(deep=True)
                    out.metadata.lmnr = LaminarEvalMetadata(
                        eval_id=self.metadata.lmnr.eval_id,
                        datapoint_id=(
                            pending_info.datapoint_id if pending_info else None
                        ),
                    )

                    await on_result(instance, out)
                    attempt_outputs.append(out)
                except asyncio.CancelledError:
                    # Task was cancelled due to timeout, error already handled
                    pass
                except Exception as e:
                    logger.error(
                        f"Unexpected error from task: {str(e)[:50]}",
                        exc_info=True,
                        stack_info=True,
                    )
                    # Create error output so the instance is not silently lost
                    if pending_info:
                        error_output = self._create_error_output(
                            pending_info.instance, e, attempt
                        )
                        if error_output.metadata is None:
                            error_output.metadata = self.metadata.model_copy(deep=True)
                        if self.metadata.lmnr is not None:
                            error_output.metadata.lmnr = LaminarEvalMetadata(
                                eval_id=self.metadata.lmnr.eval_id,
                                datapoint_id=pending_info.datapoint_id,
                            )
                        await on_result(pending_info.instance, error_output)
                        attempt_outputs.append(error_output)

            # Check for per-instance timeouts (only active tasks have start_time set)
            now = time.monotonic()
            timed_out_tasks = []
            for task in pending:
                pending_info = pending_instances.get(task)
                if (
                    pending_info
                    and pending_info.start_time
                    and now - pending_info.start_time > self.instance_timeout
                ):
                    timed_out_tasks.append(task)

            for task in timed_out_tasks:
                pending.discard(task)
                progress.update(1)
                pending_info = pending_instances.get(task)
                if pending_info:
                    inst = pending_info.instance
                    logger.error(
                        f"Instance {inst.id} timed out after {self.instance_timeout}s"
                    )
                    error_output = self._create_error_output(
                        inst,
                        TimeoutError(
                            f"Instance did not complete within "
                            f"{self.instance_timeout}s timeout"
                        ),
                        attempt,
                    )
                    if error_output.metadata is None:
                        error_output.metadata = self.metadata.model_copy(deep=True)
                    if self.metadata.lmnr is not None:
                        error_output.metadata.lmnr = LaminarEvalMetadata(
                            eval_id=self.metadata.lmnr.eval_id,
                            datapoint_id=pending_info.datapoint_id,
                        )
                    await on_result(inst, error_output)
                    attempt_outputs.append(error_output)
                # Cancel the task (the thread will continue but result is discarded)
                task.cancel()

        progress.close()
        return attempt_outputs

    def _calculate_resource_factor(self, runtime_failure_count: int) -> int:
        """Calculate the resource factor based on runtime failure count.

        Uses exponential backoff: base_factor * 2^runtime_failure_count
        Capped at max_resource_factor from metadata.

        Args:
            runtime_failure_count: Number of runtime failures encountered so far.

        Returns:
            The resource factor to use for this attempt.
        """
        if runtime_failure_count <= 0:
            return self.metadata.base_resource_factor

        factor = self.metadata.base_resource_factor * (2**runtime_failure_count)
        return min(factor, self.metadata.max_resource_factor)

    def _cleanup_workspace(
        self,
        workspace: RemoteWorkspace,
        instance: EvalInstance,
        *,
        capture_archive: bool = True,
    ) -> None:
        """Clean up workspace resources and optionally capture conversation archive."""
        if capture_archive:
            try:
                self._capture_conversation_archive(workspace, instance)
            except Exception as archive_error:
                logger.warning(
                    "[worker] Failed to capture conversation files for %s: %s",
                    instance.id,
                    archive_error,
                )
        try:
            workspace.__exit__(None, None, None)
            logger.debug("[worker] cleaned up workspace for id=%s", instance.id)
        except Exception as cleanup_error:
            logger.warning(
                f"[worker] Failed to cleanup workspace for {instance.id}: "
                f"{str(cleanup_error)[:50]}"
            )

    # --- Worker method (executed in thread executor) ---------------------------
    def _process_one_sync(
        self,
        instance: EvalInstance,
        critic_attempt: int,
        lmnr_session_id: str | None = None,
        lmnr_trace_metadata: dict[str, Any] | None = None,
        lmnr_datapoint_id: UUID | None = None,
    ) -> Tuple[EvalInstance, EvalOutput]:
        """Execute one instance synchronously in a thread with retry logic.

        This method runs in a thread executor via asyncio.to_thread().
        It performs all sync SDK operations (workspace creation, conversation).

        - Creates workspace in the thread
        - Handles retries within the thread
        - Tracks runtime failures and increases resource_factor exponentially
        - Ensures proper context-managed cleanup
        - Returns (instance, output) so the async caller can stream results
        """
        # Set up instance-specific logging + stdout/stderr redirect
        log_dir = os.path.join(self.metadata.eval_output_dir, "logs")

        from benchmarks.utils.worker_context import instance_context

        with instance_context(log_dir, instance.id):
            logger.info("[worker] start id=%s", instance.id)

            # Two-phase datapoint linking:
            # 1. Parent creates datapoint immediately (for UI progress tracking)
            # 2. Worker starts eval_span when work begins (accurate timeline)
            # 3. Link them via update_datapoint_trace_id
            #
            # Unlike ProcessPoolExecutor, asyncio threads share the process
            # so Laminar is already initialized. We just need to start a new
            # span and link it to the datapoint.
            eval_span = None
            try:
                eval_span = Laminar.start_active_span(
                    "Evaluation",
                    span_type="EVALUATION",  # type: ignore
                    session_id=lmnr_session_id,
                    metadata=lmnr_trace_metadata,
                )
                eval_span_ctx = Laminar.get_laminar_span_context(eval_span)

                if lmnr_datapoint_id is not None and self.metadata.lmnr is not None:
                    # OpenTelemetry trace_id is a 128-bit integer in span context
                    trace_id = UUID(int=eval_span.get_span_context().trace_id)
                    logger.info(
                        "[worker] Linking datapoint %s to trace %s for instance %s",
                        lmnr_datapoint_id,
                        trace_id,
                        instance.id,
                    )
                    try:
                        LaminarService.get().update_datapoint_trace_id(
                            eval_id=self.metadata.lmnr.eval_id,
                            datapoint_id=lmnr_datapoint_id,
                            trace_id=trace_id,
                        )
                    except Exception as exc:
                        logger.error(
                            "[worker] Failed to link datapoint %s to trace for instance %s: %s",
                            lmnr_datapoint_id,
                            instance.id,
                            exc,
                            exc_info=True,
                        )

                runtime_runs: list[RemoteRuntimeAllocation] = []
                out, _ = self._execute_single_attempt(
                    instance=instance,
                    eval_span_ctx=eval_span_ctx,
                    critic_attempt=critic_attempt,
                    resource_factor=self.metadata.base_resource_factor,
                    retry_count=0,
                    max_retries=0,
                    runtime_failure_count=0,
                    runtime_runs=runtime_runs,
                )
                assert out is not None
                return instance, out
            finally:
                if eval_span is not None:
                    _safe_end_span(eval_span, "eval_span")

    def _execute_single_attempt(
        self,
        instance: EvalInstance,
        eval_span_ctx: Any,
        critic_attempt: int,
        resource_factor: int,
        retry_count: int,
        max_retries: int,
        runtime_failure_count: int,
        runtime_runs: list[RemoteRuntimeAllocation],
    ) -> tuple[EvalOutput | None, FailureCategory | None]:
        """Execute one attempt with proper span and workspace lifecycle.

        Returns a ``(output, failure_category)`` tuple:

            ``(EvalOutput, None)``
                On success, or on the *final* retry failure so the caller can
                report it.

            ``(None, FailureCategory)``
                On a non-final failure, signalling the caller should retry.
                The category tells the caller whether to escalate
                ``resource_factor``.
        """
        workspace = None
        exec_span = None
        virtual_key: str | None = None
        conversation_archive_path: Path | None = None
        try:
            # Serialize span context and inject via environment variable so
            # workspace can pick it up. Use a lock to avoid races between
            # threads that read/write the same env-var key.
            exec_span = Laminar.start_active_span(
                "Execution",
                span_type="EXECUTOR",  # type: ignore
                parent_span_context=eval_span_ctx,
            )
            exec_span_ctx = json.dumps(Laminar.serialize_span_context(exec_span))
            with _lmnr_env_lock:
                os.environ["LMNR_SPAN_CONTEXT"] = exec_span_ctx or ""

            if runtime_failure_count > 0:
                logger.warning(
                    f"[worker] Instance {instance.id}: "
                    f"attempt {retry_count + 1}/{max_retries + 1}, "
                    f"runtime_failure_count={runtime_failure_count}, "
                    f"resource_factor={resource_factor}"
                )

            # Create a per-instance LiteLLM virtual key for exact cost tracking.
            # The key is stored in thread-local so build_acp_agent() can inject
            # it into the ACP subprocess env. No-op when LITELLM_MASTER_KEY unset.
            run_id = os.getenv("UNIQUE_EVAL_NAME")
            virtual_key = create_virtual_key(instance.id, run_id=run_id)
            set_current_virtual_key(virtual_key)

            workspace = self.prepare_workspace(
                instance,
                resource_factor=resource_factor,
                forward_env=LMNR_ENV_VARS,
            )

            # Record runtime/pod mapping only for remote runtimes
            if isinstance(workspace, APIRemoteWorkspace):
                retry_number = retry_count + 1  # 1-indexed for readability
                runtime_run = RemoteRuntimeAllocation(
                    runtime_id=getattr(workspace, "_runtime_id", None),
                    session_id=getattr(workspace, "session_id", None),
                    runtime_url=getattr(workspace, "_runtime_url", None),
                    resource_factor=resource_factor,
                    critic_attempt=critic_attempt,
                    retry=retry_number,
                    started_at=datetime.now(timezone.utc),
                )
                runtime_runs.append(runtime_run)
                logger.info(
                    "[worker] runtime allocated instance=%s attempt=%d retry=%d workspace=%s runtime_id=%s session_id=%s resource_factor=%s",
                    instance.id,
                    critic_attempt,
                    retry_number,
                    workspace.__class__.__name__,
                    runtime_run.runtime_id,
                    runtime_run.session_id,
                    runtime_run.resource_factor,
                )
            out = self.evaluate_instance(instance, workspace)
            if runtime_runs:
                out.runtime_runs = runtime_runs

            # Query exact cost from the LiteLLM proxy virtual key.
            # Stored alongside the SDK's token-count estimate so both
            # values are available in the output JSON.
            proxy_cost = self._query_proxy_cost(instance.id, virtual_key)
            if proxy_cost is not None:
                out.test_result["proxy_cost"] = proxy_cost
                logger.info(
                    "[worker] proxy cost for %s: $%.6f",
                    instance.id,
                    proxy_cost,
                )

            logger.info("[worker] done id=%s", instance.id)
            return out, None
        except Exception as e:
            if exec_span is not None:
                exec_span.record_exception(e)

            # Log structured runtime allocation/init failures so we can trace instance -> runtime/pod
            runtime_id = getattr(workspace, "_runtime_id", None) if workspace else None
            session_id = getattr(workspace, "session_id", None) if workspace else None
            if isinstance(workspace, APIRemoteWorkspace) or (
                "Runtime not yet ready" in str(e)
            ):
                logger.warning(
                    "[worker] runtime init failure instance=%s attempt=%d retry=%d runtime_id=%s session_id=%s error=%s",
                    instance.id,
                    critic_attempt,
                    retry_count + 1,
                    runtime_id,
                    session_id,
                    str(e),
                )

            # Classify the failure to decide whether resource escalation
            # is warranted.  See evaluation/issues/408.
            failure_category = classify_failure(e)
            escalate = failure_category == FailureCategory.RESOURCE

            logger.warning(
                "[worker] Instance %s: failure_category=%s, escalate_resources=%s, "
                "runtime_failure_count=%d",
                instance.id,
                failure_category.value,
                escalate,
                runtime_failure_count + (1 if escalate else 0),
            )

            if retry_count < max_retries:
                logger.warning(
                    f"[worker] Instance {instance.id} failed "
                    f"(attempt {retry_count + 1}/{max_retries + 1}): "
                    f"{str(e)}"
                )
            else:
                logger.error(
                    f"[worker] Instance {instance.id} failed after "
                    f"{max_retries + 1} attempts. Last error: {str(e)}",
                    exc_info=True,
                )
                if workspace is not None:
                    # Capture the archive before teardown so failure telemetry
                    # survives even when cleanup later skips duplicate capture.
                    conversation_archive_path = self._capture_conversation_archive(
                        workspace,
                        instance,
                    )
                recovered_metrics = self._load_metrics_from_conversation_archive(
                    conversation_archive_path
                )
                proxy_cost = self._query_proxy_cost(instance.id, virtual_key)
                error_output = self._create_error_output(
                    instance,
                    e,
                    max_retries,
                    metrics=recovered_metrics,
                    proxy_cost=proxy_cost,
                )
                if runtime_runs:
                    error_output.runtime_runs = runtime_runs
                return error_output, None
            return None, failure_category
        finally:
            # Clean up the per-instance virtual key and thread-local.
            if virtual_key is not None:
                delete_key(virtual_key)
            set_current_virtual_key(None)
            if workspace is not None:
                self._cleanup_workspace(
                    workspace,
                    instance,
                    capture_archive=conversation_archive_path is None,
                )
            if exec_span is not None:
                _safe_end_span(exec_span, "exec_span")


# ---------- Thread-safety helpers ------------------------------------------------


def _safe_end_span(span: Any, label: str) -> None:
    """End a span, handling contextvars errors from cross-thread usage.

    OpenTelemetry spans use contextvars tokens that can only be detached
    in the thread where they were attached. When a span created in the main
    thread is ended in a worker thread, LookupError is raised.
    """
    try:
        span.end()
    except LookupError:
        # Expected when span was created in main thread but ended in worker.
        # The span data is still recorded; only the context detach fails.
        pass
    except Exception as e:
        logger.warning(
            "[worker] %s.end() unexpected error (%s): %s", label, type(e).__name__, e
        )


# Lock to serialise writes to os.environ["LMNR_SPAN_CONTEXT"].
# The env-var is read by prepare_workspace(); the lock ensures the value set by
# one thread isn't overwritten by another before the workspace picks it up.
_lmnr_env_lock = threading.Lock()
