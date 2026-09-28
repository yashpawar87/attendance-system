#!/usr/bin/env python3
"""Enroll ChokePoint subjects from XML-selected frames.

The XML is used only as ground-truth metadata: it tells this script which
frames belong to which synthetic subject. The XML eye coordinates are never
passed to the recognition pipeline. Every selected frame is independently
detected by YuNet and embedded by SFace. If an extracted JPEG is unavailable,
the frame is read from the first MP4 in the dataset directory and cached as
that frame's JPEG.

Example:
    python enroll_chokepoint.py \
        --dataset-dir create_syn_data/P1E_S1_C1 \
        --xml create_syn_data/P1E_S1_C1.xml \
        --count 5 \
        --manifest-out create_syn_data/P1E_S1_C1/enrollment_manifest.json

Use --dry-run to validate selection/model output without inserting rows.
Use --replace to explicitly replace existing embeddings for the mapped people.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameAnnotation:
    subject_id: str
    frame_number: int
    left_eye: tuple[float, float] | None
    right_eye: tuple[float, float] | None


@dataclass(frozen=True)
class SubjectFrames:
    subject_id: str
    annotations: tuple[FrameAnnotation, ...]

    @property
    def frame_numbers(self) -> tuple[int, ...]:
        return tuple(annotation.frame_number for annotation in self.annotations)


@dataclass(frozen=True)
class SelectedFrame:
    subject_id: str
    frame_number: int
    path: Path
    confidence: float
    face_area_ratio: float
    annotation: FrameAnnotation


def parse_ground_truth(xml_path: Path) -> list[SubjectFrames]:
    """Return subject/frame membership in first-appearance order.

    Frames with no person annotation are ignored. A frame containing multiple
    annotated people is also ignored because this enrollment script must not
    guess which XML identity corresponds to a detected face. Eye coordinates
    are retained only to associate the correct YuNet detection in a frame that
    contains another visible person; they are never used for alignment.
    """
    root = ET.parse(xml_path).getroot()
    by_subject: dict[str, list[FrameAnnotation]] = {}
    for frame in root.findall("frame"):
        frame_number = int(frame.attrib["number"])
        people = frame.findall("person")
        if len(people) != 1:
            continue
        person = people[0]
        subject_id = person.attrib["id"]

        def eye_position(name: str) -> tuple[float, float] | None:
            eye = person.find(name)
            if eye is None or "x" not in eye.attrib or "y" not in eye.attrib:
                return None
            return float(eye.attrib["x"]), float(eye.attrib["y"])

        annotation = FrameAnnotation(subject_id, frame_number, eye_position("leftEye"), eye_position("rightEye"))
        by_subject.setdefault(subject_id, []).append(annotation)
    return [SubjectFrames(subject_id, tuple(annotations)) for subject_id, annotations in by_subject.items()]


def load_subject_mapping(path: Path | None, subjects: Iterable[SubjectFrames], prefix: str) -> dict[str, str]:
    subject_ids = [subject.subject_id for subject in subjects]
    if path is None:
        # XML first appearance is intentional: subject 0003 is EMP001 for this
        # P1E_S1_C1 file, subject 0005 is EMP002, and so on.
        return {subject_id: f"{prefix}{index:03d}" for index, subject_id in enumerate(subject_ids, start=1)}
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError("Mapping JSON must be an object such as {\"0003\": \"EMP001\"}")
    result = {str(subject_id): str(employee_code) for subject_id, employee_code in mapping.items()}
    missing = [subject_id for subject_id in subject_ids if subject_id not in result]
    if missing:
        raise ValueError(f"Mapping is missing XML subjects: {', '.join(missing)}")
    return {subject_id: result[subject_id] for subject_id in subject_ids}


def frame_path(dataset_dir: Path, frame_number: int) -> Path:
    padded = dataset_dir / f"{frame_number:08d}.jpg"
    if padded.exists():
        return padded
    unpadded = dataset_dir / f"{frame_number}.jpg"
    return unpadded


def video_path(dataset_dir: Path) -> Path | None:
    videos = sorted(dataset_dir.glob("*.mp4")) + sorted(dataset_dir.glob("*.MP4"))
    return videos[0] if videos else None


def load_frame(
    image_path: Path,
    frame_number: int,
    video_capture: cv2.VideoCapture | None,
    video_frame_number: int | None = None,
) -> np.ndarray | None:
    if image_path.exists():
        return cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if video_capture is None:
        return None
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame_number if video_frame_number is not None else frame_number)
    ok, image = video_capture.read()
    if not ok or image is None:
        return None
    # Cache only selected frames. The extracted JPEGs are ignored by Git,
    # keeping deployments small while preserving the existing manifest paths.
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Could not cache video frame: {image_path}")
    return image


def spread_order(frame_numbers: tuple[int, ...], count: int) -> list[int]:
    """Return candidates nearest evenly spaced points across the appearance."""
    if not frame_numbers:
        return []
    target_count = min(count, len(frame_numbers))
    targets = np.linspace(0, len(frame_numbers) - 1, target_count)
    ordered: list[int] = []
    used: set[int] = set()
    for target in targets:
        index = int(round(float(target)))
        for candidate_index in sorted(range(len(frame_numbers)), key=lambda value: abs(value - index)):
            candidate = frame_numbers[candidate_index]
            if candidate not in used:
                ordered.append(candidate)
                used.add(candidate)
                break
    # If quality filtering rejects a target, search the remaining frames in
    # temporal order rather than silently enrolling fewer samples.
    ordered.extend(frame for frame in frame_numbers if frame not in used)
    return ordered


def detect_one(detector, image: np.ndarray, annotation: FrameAnnotation | None = None):
    if hasattr(detector, "detect_all"):
        faces = detector.detect_all(image)
        if not faces:
            return None
        if annotation is not None and annotation.left_eye is not None and annotation.right_eye is not None:
            expected = np.asarray([annotation.left_eye, annotation.right_eye], dtype=np.float32)
            scored = []
            for face in faces:
                detected = np.asarray(face.landmarks[:2], dtype=np.float32)
                same_order = np.linalg.norm(detected - expected, axis=1).mean()
                swapped_order = np.linalg.norm(detected - expected[::-1], axis=1).mean()
                scored.append((min(float(same_order), float(swapped_order)), face))
            distance, best = min(scored, key=lambda item: item[0])
            max_distance = max(20.0, 0.75 * float(np.sqrt(best.box[2] * best.box[3])))
            if distance <= max_distance:
                return best
            # Video decoding can slightly shift landmark coordinates compared
            # with the extracted JPEG/XML pair. If YuNet found exactly one
            # face, there is no identity ambiguity, so use that detection.
            return faces[0] if len(faces) == 1 else None
        if len(faces) != 1:
            return None
        return faces[0]
    return detector.detect(image)


def select_frames(
    subjects: list[SubjectFrames],
    dataset_dir: Path,
    detector,
    *,
    count: int,
    min_face_area_ratio: float,
    source_video: Path | None = None,
    video_frame_indices: dict[int, int] | None = None,
) -> tuple[list[SelectedFrame], dict[str, list[str]]]:
    selected: list[SelectedFrame] = []
    rejected: dict[str, list[str]] = {}
    capture = cv2.VideoCapture(str(source_video)) if source_video is not None else None
    try:
        if capture is not None and not capture.isOpened():
            capture.release()
            capture = None
        for subject in subjects:
            accepted_for_subject: list[SelectedFrame] = []
            rejected_for_subject: list[str] = []
            annotations_by_frame = {annotation.frame_number: annotation for annotation in subject.annotations}
            for frame_number in spread_order(subject.frame_numbers, count):
                image_path = frame_path(dataset_dir, frame_number)
                image = load_frame(image_path, frame_number, capture, (video_frame_indices or {}).get(frame_number))
                if image is None:
                    reason = "missing_image" if source_video is None else "missing_image_and_video_frame"
                    rejected_for_subject.append(f"{frame_number}:{reason}")
                    continue
                annotation = annotations_by_frame[frame_number]
                face = detect_one(detector, image, annotation)
                if face is None:
                    rejected_for_subject.append(f"{frame_number}:no_yunet_face_associated_to_annotation")
                    continue
                _, _, width, height = face.box
                area_ratio = float(width * height) / float(image.shape[0] * image.shape[1])
                if area_ratio < min_face_area_ratio:
                    rejected_for_subject.append(f"{frame_number}:face_too_small")
                    continue
                accepted_for_subject.append(SelectedFrame(subject.subject_id, frame_number, image_path, face.confidence, area_ratio, annotation))
                if len(accepted_for_subject) == count:
                    break
            selected.extend(accepted_for_subject)
            rejected[subject.subject_id] = rejected_for_subject
            if len(accepted_for_subject) < count:
                raise RuntimeError(
                    f"Subject {subject.subject_id} has only {len(accepted_for_subject)} usable frames; "
                    f"requested {count}. Rejected: {', '.join(rejected_for_subject[:8])}"
                )
    finally:
        if capture is not None:
            capture.release()
    return selected, rejected


def build_embeddings(selected: list[SelectedFrame], detector, embedder) -> dict[str, list[tuple[SelectedFrame, list[float]]]]:
    embeddings: dict[str, list[tuple[SelectedFrame, list[float]]]] = {}
    for item in selected:
        image = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode selected image: {item.path}")
        face = detect_one(detector, image, item.annotation)
        if face is None:
            raise RuntimeError(f"YuNet could not associate a face with the selected annotation: {item.path}")
        # The detector's result is used for alignment and embedding. XML eye
        # coordinates are used only above to select among detections; they are
        # deliberately not passed to SFace or the recognition code.
        face_with_image = type(face)(face.box, face.landmarks, face.confidence, image)
        vector = embedder.embed(face_with_image)
        if len(vector) != 128:
            raise RuntimeError(f"SFace returned {len(vector)} values for {item.path}; expected 128")
        embeddings.setdefault(item.subject_id, []).append((item, vector))
    return embeddings


def write_manifest(path: Path, *, xml_path: Path, dataset_dir: Path, mapping: dict[str, str], embeddings) -> None:
    payload = {
        "dataset": dataset_dir.name,
        "xml": str(xml_path),
        "identity_source": "XML subject/frame membership; XML eyes only associate the selected YuNet detection",
        "xml_eye_coordinates_used_to_select_yunet_detection": True,
        "xml_eye_coordinates_used_for_embedding": False,
        "subjects": {
            subject_id: {
                "employee_code": employee_code,
                "frames": [item.frame_number for item, _ in embeddings.get(subject_id, [])],
                "files": [str(item.path) for item, _ in embeddings.get(subject_id, [])],
            }
            for subject_id, employee_code in mapping.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def insert_embeddings(
    mapping: dict[str, str],
    embeddings,
    *,
    replace: bool,
    name_prefix: str,
    employee_prefix: str,
) -> int:
    from sqlalchemy import delete, select
    from sqlalchemy.orm import Session

    from app.db.database import get_engine
    from app.db.models import FaceEmbedding, Person

    db = Session(get_engine())
    inserted = 0
    try:
        for subject_id, employee_code in mapping.items():
            person = db.scalar(select(Person).where(Person.employee_code == employee_code))
            if person is not None and not replace:
                raise RuntimeError(
                    f"{employee_code} already exists; use --replace to replace its embeddings explicitly"
                )
            if person is None:
                number = employee_code.removeprefix(employee_prefix)
                person = Person(employee_code=employee_code, name=f"{name_prefix} {number}", active=True)
                db.add(person)
                db.flush()
            if replace:
                db.execute(delete(FaceEmbedding).where(FaceEmbedding.person_id == person.id))
            first_selected = embeddings[subject_id][0][0]
            person.profile_photo = first_selected.path.read_bytes()
            person.profile_photo_content_type = "image/jpeg"
            rows = [
                FaceEmbedding(person_id=person.id, embedding=vector, model_name="sface", model_version="2021dec")
                for _, vector in embeddings[subject_id]
            ]
            db.add_all(rows)
            inserted += len(rows)
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll ChokePoint identities with YuNet + SFace")
    parser.add_argument("--dataset-dir", type=Path, default=Path("create_syn_data/P1E_S1_C1"))
    parser.add_argument("--xml", type=Path, default=Path("create_syn_data/P1E_S1_C1.xml"))
    parser.add_argument("--model-dir", type=Path, default=Path("app/vision/models"))
    parser.add_argument("--mapping-file", type=Path, help="JSON mapping such as {\"0003\": \"EMP001\"}")
    parser.add_argument("--employee-prefix", default="EMP")
    parser.add_argument("--name-prefix", default="Employee")
    parser.add_argument("--count", type=int, default=5, help="Usable enrollment frames per subject")
    parser.add_argument("--min-face-area-ratio", type=float, default=0.01)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Run detection/embedding but do not write PostgreSQL rows")
    parser.add_argument("--replace", action="store_true", help="Replace existing embeddings for mapped employees")
    parser.add_argument("--database-url", help="Optional PostgreSQL URL; otherwise DATABASE_URL/.env is used")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")
    if not args.xml.is_file():
        raise SystemExit(f"Ground-truth XML not found: {args.xml}")
    model_paths = {
        "detection": args.model_dir / "face_detection_yunet_2023mar.onnx",
        "embedding": args.model_dir / "face_recognition_sface_2021dec.onnx",
    }
    missing_models = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing_models:
        raise SystemExit("Missing model files:\n  " + "\n  ".join(missing_models))

    from app.vision.detector import YuNetFaceDetector
    from app.vision.embedder import SFaceEmbedder

    subjects = parse_ground_truth(args.xml)
    if not subjects:
        raise SystemExit("The XML contains no single-person frame annotations")
    mapping = load_subject_mapping(args.mapping_file, subjects, args.employee_prefix)
    print(f"Found {len(subjects)} subjects in {args.xml}")
    print("Mapping: " + ", ".join(f"{subject}->{employee}" for subject, employee in mapping.items()))
    source_video = video_path(args.dataset_dir)
    video_frame_indices = {
        int(frame.attrib["number"]): index
        for index, frame in enumerate(ET.parse(args.xml).getroot().findall("frame"))
    }
    if source_video is not None:
        print(f"Frame source fallback: {source_video}")

    detector = YuNetFaceDetector(model_paths["detection"])
    embedder = SFaceEmbedder(model_paths["embedding"])
    selected, rejected = select_frames(
        subjects,
        args.dataset_dir,
        detector,
        count=args.count,
        min_face_area_ratio=args.min_face_area_ratio,
        source_video=source_video,
        video_frame_indices=video_frame_indices,
    )
    embeddings = build_embeddings(selected, detector, embedder)

    for subject in subjects:
        print(f"{subject.subject_id} -> {mapping[subject.subject_id]}: {len(embeddings[subject.subject_id])} embeddings")
    rejected_count = sum(len(items) for items in rejected.values())
    if rejected_count:
        print(f"Skipped {rejected_count} unusable candidate frames", file=sys.stderr)

    if args.manifest_out:
        write_manifest(args.manifest_out, xml_path=args.xml, dataset_dir=args.dataset_dir, mapping=mapping, embeddings=embeddings)
        print(f"Wrote enrollment manifest: {args.manifest_out}")
    if args.dry_run:
        print("Dry run complete; no PostgreSQL rows were written.")
        return 0
    inserted = insert_embeddings(
        mapping,
        embeddings,
        replace=args.replace,
        name_prefix=args.name_prefix,
        employee_prefix=args.employee_prefix,
    )
    print(f"Inserted {inserted} embeddings into PostgreSQL/pgvector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
