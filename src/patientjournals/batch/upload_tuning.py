from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


UploadProfileName = Literal["light", "normal", "aggressive"]


@dataclass(frozen=True)
class UploadProfile:
    min_workers: int
    max_workers: int
    up_step: int
    down_step: int
    batch_multiplier: int
    min_batch_limit: int
    speedup_ratio: float
    slowdown_ratio: float


_PROFILES: dict[UploadProfileName, UploadProfile] = {
    "light": UploadProfile(
        min_workers=1,
        max_workers=8,
        up_step=1,
        down_step=1,
        batch_multiplier=2,
        min_batch_limit=4,
        speedup_ratio=1.10,
        slowdown_ratio=0.78,
    ),
    "normal": UploadProfile(
        min_workers=2,
        max_workers=24,
        up_step=2,
        down_step=2,
        batch_multiplier=3,
        min_batch_limit=8,
        speedup_ratio=1.08,
        slowdown_ratio=0.82,
    ),
    "aggressive": UploadProfile(
        min_workers=4,
        max_workers=64,
        up_step=4,
        down_step=4,
        batch_multiplier=4,
        min_batch_limit=16,
        speedup_ratio=1.05,
        slowdown_ratio=0.86,
    ),
}


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


class UploadAutoTuner:
    def __init__(
        self,
        *,
        profile: UploadProfileName,
        initial_workers: int,
        initial_batch_limit: int,
        max_workers_override: int = 0,
    ) -> None:
        if profile not in _PROFILES:
            raise ValueError(
                f"Unsupported upload profile: {profile}. "
                f"Expected one of: {sorted(_PROFILES)}"
            )
        self.profile_name = profile
        self.profile = _PROFILES[profile]
        hard_max_workers = self.profile.max_workers
        if max_workers_override > 0:
            hard_max_workers = min(hard_max_workers, int(max_workers_override))
        self.max_workers = max(self.profile.min_workers, hard_max_workers)

        start_workers = _clamp(
            int(initial_workers or self.profile.min_workers),
            self.profile.min_workers,
            self.max_workers,
        )
        self.current_workers = start_workers
        base_batch = max(
            int(initial_batch_limit or 0),
            self.current_workers * self.profile.batch_multiplier,
            self.profile.min_batch_limit,
        )
        self.current_batch_limit = base_batch
        self._last_rate: float | None = None

    def record_batch(self, items: int, seconds: float, *, had_errors: bool = False) -> None:
        if items <= 0 or seconds <= 0:
            return
        rate = items / seconds
        if had_errors:
            self.current_workers = max(
                self.profile.min_workers,
                self.current_workers - self.profile.down_step,
            )
            self._sync_batch_limit()
            self._last_rate = rate
            return

        previous = self._last_rate
        if previous is not None:
            if rate >= previous * self.profile.speedup_ratio:
                self.current_workers = min(
                    self.max_workers,
                    self.current_workers + self.profile.up_step,
                )
            elif rate <= previous * self.profile.slowdown_ratio:
                self.current_workers = max(
                    self.profile.min_workers,
                    self.current_workers - self.profile.down_step,
                )

        self._sync_batch_limit()
        self._last_rate = rate

    def _sync_batch_limit(self) -> None:
        self.current_batch_limit = max(
            self.profile.min_batch_limit,
            self.current_workers * self.profile.batch_multiplier,
        )


def build_upload_tuner(
    *,
    profile: str,
    initial_workers: int,
    initial_batch_limit: int,
    max_workers_override: int = 0,
) -> UploadAutoTuner:
    normalized = str(profile or "normal").strip().lower()
    if normalized not in _PROFILES:
        raise ValueError(
            f"Unsupported upload profile: {profile}. "
            f"Expected one of: {sorted(_PROFILES)}"
        )
    return UploadAutoTuner(
        profile=normalized,  # type: ignore[arg-type]
        initial_workers=initial_workers,
        initial_batch_limit=initial_batch_limit,
        max_workers_override=max_workers_override,
    )

