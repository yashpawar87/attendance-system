from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.repositories import find_best_matches
from app.vision.interfaces import FaceMatcher, MatchCandidate


class PgVectorFaceMatcher(FaceMatcher):
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings

    def match(self, embedding: list[float]) -> tuple[MatchCandidate | None, MatchCandidate | None]:
        matches = find_best_matches(self.db, embedding, limit=2)
        candidates = [MatchCandidate(person_id=person_id, similarity=similarity) for person_id, similarity in matches]
        return (candidates + [None, None])[:2]
