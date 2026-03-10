import math
import random
import bisect


class TrafficGenerator:
    def __init__(
        self,
        num_urls: int = 20,
        total_requests: int = 1000,
        alpha: float = 1.2,
        seed: int | None = 42,
    ):
        if num_urls < 1:
            raise ValueError("num_urls must be >= 1")
        if total_requests < 1:
            raise ValueError("total_requests must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha (Zipf skew) must be > 0")

        self.num_urls = num_urls
        self.total_requests = total_requests
        self.alpha = alpha
        self._rng = random.Random(seed)

        # Pre-compute the Zipf Cumulative Distribution Function (CDF)
        # weights[k] = 1 / (k+1)^alpha  for k in 0..num_urls-1        O(num_urls)
        weights = [1.0 / (k + 1) ** alpha for k in range(num_urls)]
        total_weight = sum(weights)

        # Normalised CDF — used for O(log n) inverse-transform sampling
        cumulative = 0.0
        self._cdf: list[float] = []
        for w in weights:
            cumulative += w / total_weight
            self._cdf.append(cumulative)

        # URL labels (rank 0 = most popular)
        self._urls = [f"url_{k + 1}" for k in range(num_urls)]

    def generate(self) -> list[str]:
        requests: list[str] = []
        for _ in range(self.total_requests):
            r = self._rng.random()                      # uniform [0, 1)
            idx = bisect.bisect_left(self._cdf, r)      # O(log num_urls)
            idx = min(idx, self.num_urls - 1)           # clamp for float precision
            requests.append(self._urls[idx])
        return requests

    def url_popularity_table(self) -> list[tuple[str, float]]:
        probs = []
        total_weight = sum(1.0 / (k + 1) ** self.alpha for k in range(self.num_urls))
        for k in range(self.num_urls):
            prob = (1.0 / (k + 1) ** self.alpha) / total_weight
            probs.append((self._urls[k], round(prob * 100, 2)))
        return probs
