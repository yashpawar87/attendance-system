#!/usr/bin/env python3
"""Populate dashboard profile photos from an existing ChokePoint manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_engine
from app.db.models import Person


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate employee profile photos from enrollment_manifest.json")
    parser.add_argument("--manifest", type=Path, default=Path("create_syn_data/P1E_S1_C1/enrollment_manifest.json"))
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    updated = 0
    skipped = 0
    with Session(get_engine()) as db:
        for subject_id, entry in payload.get("subjects", {}).items():
            employee_code = str(entry["employee_code"])
            files = [Path(item) for item in entry.get("files", [])]
            photo_path = next((path for path in files if path.is_file()), None)
            person = db.scalar(select(Person).where(Person.employee_code == employee_code))
            if person is None or photo_path is None:
                skipped += 1
                continue
            person.profile_photo = photo_path.read_bytes()
            person.profile_photo_content_type = "image/jpeg"
            updated += 1
        db.commit()
    print(f"Updated {updated} employee profile photos; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
