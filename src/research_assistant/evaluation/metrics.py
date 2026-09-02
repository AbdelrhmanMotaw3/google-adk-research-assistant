from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import statistics


@dataclass
class Metrics:
    queries: int = 0
    quality_scores: List[float] = field(default_factory=list)
    sources_used: List[int] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    credibility: List[float] = field(default_factory=list)
    citations: List[int] = field(default_factory=list)
    wall_times_s: List[float] = field(default_factory=list)

    def record(self, *, quality: float, sources: int, iters: int, credibility: float, citations: int, wall_time_s: float) -> None:
        self.queries += 1
        self.quality_scores.append(float(quality))
        self.sources_used.append(int(sources))
        self.iterations.append(int(iters))
        self.credibility.append(float(credibility))
        self.citations.append(int(citations))
        self.wall_times_s.append(float(wall_time_s))

    def summary(self) -> Dict[str, Any]:
        def avg(xs: List[float]) -> float:
            return float(statistics.mean(xs)) if xs else 0.0

        return {
            "queries": self.queries,
            "avg_quality": avg(self.quality_scores),
            "avg_sources": avg([float(x) for x in self.sources_used]),
            "avg_iterations": avg([float(x) for x in self.iterations]),
            "avg_credibility": avg(self.credibility),
            "avg_citations": avg([float(x) for x in self.citations]),
            "avg_wall_time_s": avg(self.wall_times_s),
        }

