# Door-based face recognition attendance

This repository implements the PRD as a small monorepo:

- `app/` is the FastAPI backend and owns Postgres/pgvector access.
- `dashboard/` is a stateless Node/Express reporting UI that calls only the backend REST API.
- `door/` is the on-premise camera client and is not deployed to Railway.

## Run locally

```bash
cp .env.example .env
docker compose up -d db
python -m pip install -e '.[test]'
alembic upgrade head
uvicorn app.main:app --reload
```

Put the YuNet, SFace, and MiniFASNetV2 ONNX weights in `app/vision/models/` before using recognition or enrollment. The backend returns a clear `503` when weights are absent; it never substitutes a different inference runtime or model.

Run the dashboard from its directory with `npm install`, `BACKEND_URL=http://localhost:8000 API_TOKEN=change-me CAMERA_SOURCE_ID=door-camera LOCAL_CONTROL_ENABLED=true npm start`. The dashboard can use the browser camera directly, or receive frames from the on-premise camera client.

All protected endpoints use `Authorization: Bearer $API_TOKEN`. Configure thresholds and the local attendance timezone through environment variables. Attendance timestamps are stored in UTC while `attendance_date` is derived in `ATTENDANCE_TIMEZONE`, and duplicate check-ins are guarded by a database uniqueness constraint.

Uploaded frames are decoded and processed in memory only; they are never written to disk or logs. Face embeddings remain while an employee is enrolled and can be removed with the authenticated `DELETE /api/v1/people/{id}/embeddings` endpoint. Administrative actions emit audit log records containing the action, target, and actor, never biometric payloads.

The bundled MiniFASNetV2 export returns `[live, print-attack, replay-attack]`; keep `LIVENESS_LIVE_CLASS_INDEX=0`. A prerecorded MP4 is correctly treated as a replay and will not create attendance rows. Use a real camera for an end-to-end liveness/attendance test.

## Enroll the synthetic ChokePoint sequence

For `P1E_S1_C1`, `enroll_chokepoint.py` reads `P1E_S1_C1.xml` to associate frame numbers with synthetic subject IDs. When another person is visible in the same frame, the XML eye positions are used only to select the corresponding YuNet detection; the XML points are never used as SFace landmarks. Each selected JPEG is independently detected by YuNet, aligned from YuNet landmarks, embedded with SFace, and then inserted into PostgreSQL/pgvector. The default first-appearance mapping makes subject `0003` become `EMP001`, subject `0005` become `EMP002`, and so on.

After the database migration and model files are ready, run:

```bash
python enroll_chokepoint.py \
  --dataset-dir create_syn_data/P1E_S1_C1 \
  --xml create_syn_data/P1E_S1_C1.xml \
  --count 5 \
  --manifest-out create_syn_data/P1E_S1_C1/enrollment_manifest.json
```

Use `--dry-run` first to run YuNet/SFace and write the manifest without database inserts. If you intentionally need to regenerate an existing employee’s embeddings, add `--replace`; otherwise rerunning fails rather than silently duplicating data. The manifest records the selected frames and mapping for later comparison against XML labels during video accuracy evaluation; those labels are never passed into recognition.

## Demonstrate with the ChokePoint video

After enrollment, run the backend and dashboard in separate terminals, then replay the MP4 as if it were the door camera:

```bash
# Terminal 1
source .venv/bin/activate
uvicorn app.main:app --reload
```

```bash
# Terminal 2
cd dashboard
BACKEND_URL=http://localhost:8000 API_TOKEN=change-me CAMERA_SOURCE_ID=door-camera LOCAL_CONTROL_ENABLED=true npm start
```

```bash
# Terminal 3, from the repository root
source .venv/bin/activate
python door/main.py \
  --video /Users/yashpawar/attendance-system/create_syn_data/P1E_S1_C1/chokepoint_P1E_S1_C1.mp4 \
  --api-url http://localhost:8000 \
  --token change-me \
  --source-id door-camera \
  --sample-fps 5
```

Open `http://localhost:3000` for the dashboard—not port `8000`, which is the backend API. The camera panel receives the replayed frames and refreshes every five seconds.

Select **Employee database** at the top of the dashboard to view employee IDs, names, email addresses, active status, enrolled sample counts, and representative enrollment photos. After applying the migration, populate photos for the existing ChokePoint enrollment with:

```bash
alembic upgrade head
python populate_profile_photos.py
```

The themed dashboard also provides an **Add employee** form. It creates the employee, uploads an optional profile photo, and enrolls one or more face images through the existing protected people API. The live camera panel overlays green boxes for recognized employees and red boxes for unknown faces, and the Overview page counts distinct unknown-person appearances as **Intruders detected**. The counter is in-memory and resets when the backend restarts. On the Overview page, **Open camera** requests camera permission from the browser and sends frames securely through the dashboard to the backend. **Run video demo** plays the bundled ChokePoint MP4 in the browser and sends its frames through the explicit replay-demo endpoint. Set `DEMO_REPLAY_ENABLED=true` before starting the backend for the browser demo. Browser camera access requires HTTPS or `localhost`; Railway provides HTTPS automatically.

For a client presentation where the MP4 also creates visibly labeled demo rows, add `DEMO_REPLAY_ENABLED=true` to `.env`, restart the backend, and run:

```bash
python door/main.py \
  --video /Users/yashpawar/attendance-system/create_syn_data/P1E_S1_C1/chokepoint_P1E_S1_C1.mp4 \
  --api-url http://localhost:8000 \
  --token change-me \
  --source-id door-camera \
  --sample-fps 5 \
  --demo-replay
```

This explicit demo mode still runs YuNet and SFace against the enrolled database, but does not claim liveness or ambiguity protection. Rows appear as `demo_replay` and retain the measured liveness score, making the distinction visible in the dashboard. Never enable it for a production door. With the normal command, the prerecorded MP4 is correctly treated as a replay attack and should create no attendance rows. A real camera should use the normal command and requires three consecutive live matches before one `check_in` row is created.

This is a functional demonstration, not an unbiased accuracy score, because the same video supplies both enrollment frames and demo frames. For accuracy evaluation, compare predictions against the XML subject labels and exclude the enrollment frames recorded in `enrollment_manifest.json`.

The backend Railway configuration uses Railpack and runs `alembic upgrade head` before Uvicorn. No Dockerfile is used.

## Railway deployment

Deploy this repository as two Railway services from the same GitHub repository:

1. **Backend service** — repository root `/`, using the root `railway.json`.
2. **Dashboard service** — root directory `/dashboard`, using `dashboard/railway.json`.

Backend variables:

```text
DATABASE_URL=<Railway PostgreSQL connection string>
API_TOKEN=<strong shared secret>
CORS_ORIGINS=https://<dashboard-public-domain>
MODEL_DIR=app/vision/models
DEMO_REPLAY_ENABLED=false
ATTENDANCE_TIMEZONE=Asia/Kolkata
```

The backend accepts Railway's `postgres://` and `postgresql://` URLs and normalizes them to the installed psycopg driver. Use a Railway PostgreSQL instance with the `vector` extension available; the migration enables it automatically.

Dashboard variables:

```text
BACKEND_URL=https://<backend-public-domain>
API_TOKEN=<same backend secret>
CAMERA_SOURCE_ID=door-camera
LOCAL_CONTROL_ENABLED=false
```

The browser camera and browser MP4 demo work on the hosted dashboard. `LOCAL_CONTROL_ENABLED` only controls the optional legacy buttons that launch a Python `door` process on the same machine as the dashboard; keep it `false` on Railway. For a separate physical camera device, run the `door` client on that device and point it at the Railway backend.
