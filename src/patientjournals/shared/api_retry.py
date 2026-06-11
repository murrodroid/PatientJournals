import random

from patientjournals.config import config


def is_fatal_api_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    fatal_markers = (
        "token limit",
        "quota",
        "permission",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "billing",
    )
    return any(marker in text for marker in fatal_markers)


def is_retryable_api_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    retryable_markers = (
        "503",
        "unavailable",
        "resource_exhausted",
        "rate limit",
        "429",
        "internal",
        "deadline",
        "timed out",
        "timeout",
        "connection reset",
        "connection error",
        "temporarily",
        "backend error",
    )
    return any(marker in text for marker in retryable_markers)


def retry_delay_seconds(attempt: int) -> float:
    initial = max(0.0, float(config.api_retry_initial_delay_seconds))
    maximum = max(initial, float(config.api_retry_max_delay_seconds))
    jitter = max(0.0, float(config.api_retry_jitter_seconds))

    delay = min(initial * (2 ** max(0, attempt - 1)), maximum)
    if jitter > 0:
        delay += random.uniform(0, jitter)
    return delay
