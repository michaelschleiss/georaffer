"""Runtime utilities for long-running operations.

Provides interrupt handling, retry policies with jitter, and executor management.
"""

import contextlib
import multiprocessing
import random
import signal
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional


class InterruptManager:
    """Manages interrupt state across modules using singleton pattern.

    Usage:
        # Check if interrupted
        if InterruptManager.get().is_set():
            return

        # Signal interrupt (from signal handler)
        InterruptManager.get().signal()

        # Clear for new run
        InterruptManager.get().clear()
    """

    _instance: Optional["InterruptManager"] = None

    def __init__(self) -> None:
        self._event = threading.Event()

    @classmethod
    def get(cls) -> "InterruptManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def signal(self) -> None:
        """Signal that an interrupt has occurred."""
        self._event.set()

    def is_set(self) -> bool:
        """Check if interrupt has been signaled."""
        return self._event.is_set()

    def clear(self) -> None:
        """Clear interrupt state for new run."""
        self._event.clear()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for interrupt signal."""
        return self._event.wait(timeout)


@dataclass
class RetryPolicy:
    """Configuration for retry behavior with exponential backoff and jitter.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter_factor: Random jitter range (0.25 = +/-25%)
    """

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.25


# Default retry policy for most operations
DEFAULT_RETRY_POLICY = RetryPolicy()


def calculate_delay(attempt: int, policy: RetryPolicy) -> float:
    """Calculate delay for retry attempt with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        policy: Retry policy configuration

    Returns:
        Delay in seconds before next retry
    """
    if attempt <= 0:
        return 0.0

    # Exponential backoff capped at max_delay
    base = min(policy.base_delay * (2 ** (attempt - 1)), policy.max_delay)

    # Add jitter to avoid thundering herd
    jitter = random.uniform(1 - policy.jitter_factor, 1 + policy.jitter_factor)
    return base * jitter


def shutdown_executor(
    executor: Any, stop_event: threading.Event | None = None, grace: float = 0.2
) -> None:
    """Cancel futures and terminate workers quickly.

    Args:
        executor: ThreadPoolExecutor or ProcessPoolExecutor
        stop_event: Optional event to signal for cooperative shutdown
        grace: Grace period before forceful kill (seconds)
    """
    if executor is None:
        return

    if stop_event is not None:
        stop_event.set()

    with contextlib.suppress(Exception):
        executor.shutdown(wait=False, cancel_futures=True)

    # For ProcessPoolExecutor, forcefully terminate workers
    if isinstance(executor, ProcessPoolExecutor):
        procs = getattr(executor, "_processes", None)
        if procs:
            for p in procs.values():
                with contextlib.suppress(Exception):
                    p.terminate()

            time.sleep(grace)

            for p in procs.values():
                if p.is_alive():
                    with contextlib.suppress(Exception):
                        p.kill()

    kill_child_processes()


def kill_child_processes() -> None:
    """Best-effort kill of child processes to avoid orphans."""
    try:
        for child in multiprocessing.active_children():
            with contextlib.suppress(Exception):
                child.kill()
    except Exception:
        pass


def install_signal_handlers(
    on_interrupt: Callable[[], None],
) -> tuple[Callable | None, Callable | None]:
    """Install SIGINT and SIGTERM handlers.

    Args:
        on_interrupt: Callback to invoke on interrupt

    Returns:
        Tuple of (old_sigint_handler, old_sigterm_handler)
    """
    old_int = None
    old_term = None

    def handler(signum, frame):
        on_interrupt()

    try:
        old_int = signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        pass  # Not main thread or not supported

    with contextlib.suppress(ValueError, OSError):
        old_term = signal.signal(signal.SIGTERM, handler)

    return old_int, old_term


def install_interrupt_signal_handlers() -> tuple[Callable | None, Callable | None]:
    """Install SIGINT/SIGTERM handlers that signal the InterruptManager and raise KeyboardInterrupt."""

    def _on_interrupt() -> None:
        # Set the global interrupt flag and exit the current flow.
        InterruptManager.get().signal()
        raise KeyboardInterrupt()

    return install_signal_handlers(_on_interrupt)


def restore_signal_handlers(old_int: Callable | None, old_term: Callable | None) -> None:
    """Restore previous signal handlers.

    Args:
        old_int: Previous SIGINT handler
        old_term: Previous SIGTERM handler
    """
    if old_int is not None:
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGINT, old_int)

    if old_term is not None:
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGTERM, old_term)
