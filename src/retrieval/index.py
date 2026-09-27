from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from storage.models import RetrievalHit, RetrievalUnit

from .embeddings import Embedder


class ExactCosineIndex:
    def __init__(self, units: list[RetrievalUnit], embedder: Embedder):
        if not units:
            raise ValueError("Cannot build an index without units")
        if len({unit.unit_id for unit in units}) != len(units):
            raise ValueError("Duplicate unit_id in index")
        self.units = list(units)
        self.embedder = embedder
        self.matrix = embedder.encode([unit.text for unit in self.units])
        if self.matrix.shape[0] != len(self.units):
            raise ValueError("Embedding row count does not match unit count")
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = self.matrix / norms

    def search(
        self, query: str, top_k: int, document_id: str | None = None
    ) -> list[RetrievalHit]:
        query_vector = self.embedder.encode([query])[0]
        return self.search_vector(query_vector, top_k, document_id=document_id)

    def search_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            return []
        query_vector = query_vector / norm
        scores = self.matrix @ query_vector
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if document_id is None or self.units[index].document_id == document_id
        ]
        candidates.sort(key=lambda pair: (-pair[1], self.units[pair[0]].unit_id))
        hits: list[RetrievalHit] = []
        for rank, (index, score) in enumerate(candidates[:top_k], 1):
            unit = self.units[index]
            hits.append(
                RetrievalHit(
                    unit_id=unit.unit_id,
                    document_id=unit.document_id,
                    level=unit.level,
                    rank=rank,
                    score=score,
                    text=unit.text,
                    locators=unit.locators,
                    parent_id=unit.parent_id,
                )
            )
        return hits


@dataclass(frozen=True)
class RoutedRetriever:
    child_index: ExactCosineIndex
    parent_by_id: dict[str, RetrievalUnit]

    def retrieve(
        self,
        query: str,
        document_id: str,
        route: str,
        top_k: int,
        candidate_k: int,
    ) -> list[RetrievalHit]:
        child_hits = self.child_index.search(query, candidate_k, document_id=document_id)
        return self._route_hits(child_hits, route, top_k)

    def retrieve_vector(
        self,
        query_vector: np.ndarray,
        document_id: str,
        route: str,
        top_k: int,
        candidate_k: int,
    ) -> list[RetrievalHit]:
        child_hits = self.child_index.search_vector(
            query_vector, candidate_k, document_id=document_id
        )
        return self._route_hits(child_hits, route, top_k)

    def _route_hits(
        self,
        child_hits: list[RetrievalHit],
        route: str,
        top_k: int,
    ) -> list[RetrievalHit]:
        if route == "child":
            return _rerank(child_hits[:top_k])
        if route == "parent":
            return self._parent_hits(child_hits, top_k)
        if route == "parent_neighbor":
            return self._parent_neighbor_hits(child_hits, top_k)
        if route == "distinct_parents":
            return self._parent_hits(child_hits, max(2, top_k), require_distinct=True)
        raise ValueError(f"Unknown retrieval route: {route}")

    def _parent_hits(
        self, child_hits: list[RetrievalHit], top_k: int, require_distinct: bool = False
    ) -> list[RetrievalHit]:
        scores: dict[str, float] = {}
        for hit in child_hits:
            if not hit.parent_id:
                continue
            scores[hit.parent_id] = max(scores.get(hit.parent_id, float("-inf")), hit.score)
        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        if require_distinct:
            ordered = ordered[: max(2, top_k)]
        else:
            ordered = ordered[:top_k]
        hits: list[RetrievalHit] = []
        for rank, (parent_id, score) in enumerate(ordered, 1):
            parent = self.parent_by_id[parent_id]
            hits.append(_unit_to_hit(parent, rank, score))
        return hits

    def _parent_neighbor_hits(
        self, child_hits: list[RetrievalHit], top_k: int
    ) -> list[RetrievalHit]:
        parent_hits = self._parent_hits(child_hits, max(1, top_k))
        if not parent_hits:
            return []
        score_by_parent = {hit.unit_id: hit.score for hit in parent_hits}
        seed = self.parent_by_id[parent_hits[0].unit_id]
        for neighbor_id in seed.neighbor_ids:
            score_by_parent.setdefault(neighbor_id, parent_hits[0].score * 0.98)
        ordered = sorted(score_by_parent.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
        return [
            _unit_to_hit(self.parent_by_id[parent_id], rank, score)
            for rank, (parent_id, score) in enumerate(ordered, 1)
        ]


def _unit_to_hit(unit: RetrievalUnit, rank: int, score: float) -> RetrievalHit:
    return RetrievalHit(
        unit_id=unit.unit_id,
        document_id=unit.document_id,
        level=unit.level,
        rank=rank,
        score=score,
        text=unit.text,
        locators=unit.locators,
        parent_id=unit.parent_id,
    )


def _rerank(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(hits, 1)]
