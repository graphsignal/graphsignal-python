import time
from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def should_sample(self) -> bool:
        pass


class TimeCoordinatedSampler(Sampler):
    def __init__(self, sampling_rate: float):
        if not (0.001 <= sampling_rate <= 1000):
            raise ValueError("sampling_rate must be in [0.001, 1000].")

        self.sampling_rate = float(sampling_rate)

        self.window_ms = int(round(1000.0 / self.sampling_rate))
        self.last_window = None

    def _now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def _current_window(self) -> int:
        return self._now_ms() // self.window_ms

    def should_sample(self) -> bool:
        current_window = self._current_window()

        # Enforces one sample per time window of size 1 / sampling_rate seconds.
        if self.last_window is None or current_window != self.last_window:
            self.last_window = current_window
            return True

        return False