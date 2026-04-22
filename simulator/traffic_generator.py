"""
Traffic Generator — Zipf Distribution
=======================================
Generates a reproducible sequence of URL requests following the
Zipf (power-law) distribution, matching real CDN traffic patterns.

Zipf weight for rank k (1-indexed):  w(k) = 1 / k^alpha
Higher alpha  →  more traffic skew  →  top URLs dominate  →  more cacheable.

Time Complexity:
    __init__  : O(N log N) — CDF construction + sort
    generate  : O(n log N) — n requests via binary-search inverse-CDF
"""

from __future__ import annotations

import bisect
import random


class TrafficGenerator:
    def __init__(
        self,
        num_urls:       int   = 50,
        total_requests: int   = 1000,
        alpha:          float = 1.2,
        seed:           int | None = 42,
    ) -> None:
        if num_urls < 1:
            raise ValueError("num_urls must be >= 1")
        if total_requests < 1:
            raise ValueError("total_requests must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha (Zipf skew) must be > 0")

        self.num_urls       = num_urls
        self.total_requests = total_requests
        self.alpha          = alpha
        self._rng           = random.Random(seed)

        # Compute Zipf weights and normalised CDF             O(N)
        weights      = [1.0 / (k + 1) ** alpha for k in range(num_urls)]
        total_weight = sum(weights)

        cumulative  = 0.0
        self._cdf:  list[float] = []
        for w in weights:                                  # O(N)
            cumulative += w / total_weight
            self._cdf.append(cumulative)

        self._urls = [f"url_{k + 1}" for k in range(num_urls)]

    def generate(self) -> list[str]:
        """
        Generate total_requests URL samples via inverse-CDF sampling.
        O(n log N) — binary search per request.
        """
        requests: list[str] = []
        for _ in range(self.total_requests):               # O(n)
            r   = self._rng.random()                       # uniform [0, 1)
            idx = bisect.bisect_left(self._cdf, r)         # O(log N)
            idx = min(idx, self.num_urls - 1)
            requests.append(self._urls[idx])
        return requests

    def url_popularity_table(self) -> list[tuple[str, float]]:
        """Return URL -> theoretical probability pairs.  O(N)"""
        total = sum(1.0 / (k + 1) ** self.alpha for k in range(self.num_urls))
        return [
            (self._urls[k], round((1.0 / (k + 1) ** self.alpha) / total * 100, 2))
            for k in range(self.num_urls)
        ]
