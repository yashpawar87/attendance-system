# CLAUDE.md — Door-Based Face Recognition Attendance System

This file is the persistent context for Claude Code (or any agent) working in this
repository. Read it before making changes. It reflects the **final target
architecture**: a Python/FastAPI recognition backend, a Node.js dashboard, and a
Postgres+pgvector database, deployed to Railway using Railway's **native
Railpack builder** — **no Dockerfile anywhere in this repo.**

---

## 1. What this project is

A door-mounted camera identifies employees walking in and marks their attendance
automatically. A Node.js dashboard lets admins see today's attendance, absentees,
and history. It is **not** one giant endpoint — it's a small real-time
computer-vision system wired to a FastAPI backend, a PostgreSQL database, and a
separate reporting dashboard.

```
Camera at door
      │
      ▼
Face Detection  →  "Is there a face?"          (YuNet, ONNX)
      │
      ▼
Liveness Check  →  "Is it a real person?"      (MiniFASNetV2, ONNX)
      │
      ▼
Face Embedding  →  face → 128-d vector          (SFace, ONNX)
      │
      ▼
Face Matcher    →  vector → person_id (pgvector similarity search)
      │
      ▼
Attendance Service → duplicate policy, temporal confirmation
      │
      ▼
PostgreSQL (people, face_embeddings, attendance, doors)
      │
      ▼
Node.js Dashboard → today's attendance, absentees, search, CSV export
```

Three logical applications, one repo (monorepo):

- **`app/`** — the FastAPI + ONNX Runtime backend. Deployed to Railway as its
  own service, built by **Railpack** (Python auto-detected). No Dockerfile.
- **`dashboard/`** — a Node.js reporting UI. Deployed to Railway as its own
  service, built by **Railpack** (Node auto-detected). No Dockerfile. Talks to
  `app/` only over its REST API — it never connects to Postgres directly.
- **`door/`** — the camera client that runs on the machine physically wired to
  the door camera (a Raspberry Pi, mini-PC, etc.). **This does NOT run on
  Railway** — Railway has no camera hardware access. It runs on-prem and calls
  the Railway-hosted API over HTTPS.

---

## 2. Initial models (locked in — implement against these, not placeholders)

| Stage | Model | File | Notes |
|---|---|---|---|
| Detection | YuNet | `face_detection_yunet_2023mar.onnx` | outputs bounding box + 5-point landmarks + confidence |
| Liveness | MiniFASNetV2 | ONNX export from a Silent-Face-Anti-Spoofing–compatible implementation | binary/softmax live-vs-spoof score |
| Embedding | SFace | `face_recognition_sface_2021dec.onnx` | **128-dimensional** embedding, expects an aligned 112×112 face |

**Runtime: ONNX Runtime for all three models** (`onnxruntime.InferenceSession`).
Do not silently swap in OpenCV's high-level `cv2.FaceDetectorYN_create` /
`cv2.FaceRecognizerSF_create` wrappers for detection/embedding even though
OpenCV ships them for these exact models — the requirement here is a single
consistent inference runtime (ONNX Runtime) across detection, embedding, *and*
liveness, since liveness has no OpenCV built-in equivalent anyway. OpenCV
(`opencv-python-headless`) is still used for image I/O, color conversion, and
resizing/cropping — just not as the model-inference engine.

Preprocessing details that matter (get these wrong and accuracy silently
degrades — there's no crash to warn you):

- **YuNet** → resize per the model's expected input, run inference, decode
  boxes + 5-point landmarks.
- **SFace** requires an **aligned** 112×112 crop, not a plain bounding-box
  crop. Use the YuNet landmarks to do a similarity/affine alignment before
  embedding — this mirrors what `cv2.FaceRecognizerSF.alignCrop` does
  internally; since we're not using that wrapper, reimplement the equivalent
  alignment step explicitly in `app/vision/embedder.py`.
- **MiniFASNetV2 / Silent-Face-Anti-Spoofing** expects the face region cropped
  with the *same scale/box-expansion convention used when the ONNX model was
  exported* (the reference project commonly uses a scaled crop, e.g. ~2.7×,
  around the detection box, resized to the model's expected square input).
  A tight bounding-box crop with no scale margin will silently produce
  unreliable liveness scores.

Model weight files live in `app/vision/models/` and are committed to the repo
(YuNet + SFace + the MiniFASNetV2 export together are tens of MB, well within
normal git limits). Railpack packages whatever is in the build context
automatically — unlike a hand-written Dockerfile, there is no `COPY` step to
forget, so just make sure the files are actually committed and not
`.gitignore`d.

---

## 3. Four separate ML operations — don't conflate them

| Step | Question it answers | Interface | Concrete implementation |
|---|---|---|---|
| Detection | Is there a face in this frame? | `FaceDetector` | `YuNetFaceDetector` |
| Liveness | Is this a live person, not a photo/video/mask? | `LivenessDetector` | `MiniFASNetLivenessDetector` |
| Embedding | Convert the face into a fixed-length vector | `FaceEmbedder` | `SFaceEmbedder` |
| Matching | Which enrolled person is this vector closest to? | `FaceMatcher` | pgvector cosine-similarity query |

Never skip liveness — without it, someone holding up a photo of an employee's
face would be marked present.

Matching must never be "highest similarity wins" with no floor. Always apply a
configurable acceptance threshold, and prefer checking the **gap between the
best and second-best match**, not just the best score in isolation.

---

## 4. Repository layout (monorepo — two independently deployed services + door client)

```
attendance-system/
├── app/                          # ── Railway service "backend" ──
│   ├── main.py
│   ├── api/
│   │   ├── people.py
│   │   ├── recognition.py
│   │   └── attendance.py         # includes /today, /absentees, /export endpoints for the dashboard
│   ├── core/
│   │   ├── config.py             # env-driven settings, incl. thresholds
│   │   └── security.py           # auth for admin endpoints
│   ├── db/
│   │   ├── database.py
│   │   ├── models.py
│   │   └── repositories.py
│   ├── vision/
│   │   ├── models/                # committed .onnx weight files, see §2
│   │   │   ├── face_detection_yunet_2023mar.onnx
│   │   │   ├── face_recognition_sface_2021dec.onnx
│   │   │   └── minifasnet_v2.onnx
│   │   ├── detector.py            # YuNetFaceDetector
│   │   ├── embedder.py            # SFaceEmbedder (incl. landmark alignment)
│   │   ├── liveness.py            # MiniFASNetLivenessDetector
│   │   ├── matcher.py
│   │   └── pipeline.py
│   └── services/
│       ├── enrollment.py
│       ├── recognition_service.py
│       └── attendance_service.py
├── dashboard/                     # ── Railway service "dashboard" ──
│   ├── package.json
│   ├── src/
│   │   ├── server.js              # Express app entrypoint
│   │   ├── routes/
│   │   │   └── attendance.js      # calls backend REST API, never touches Postgres
│   │   ├── lib/
│   │   │   └── apiClient.js       # fetch wrapper around BACKEND_URL
│   │   └── views/ (or public/)    # today's attendance, absentees, search, CSV export
│   └── .env.example
├── door/                          # runs on-prem, NOT deployed to Railway
│   ├── camera.py
│   ├── client.py
│   └── main.py
├── migrations/                    # Alembic, for app/
├── tests/                         # pytest, for app/
├── railway.json                   # backend service config (builder: RAILPACK)
├── docker-compose.yml             # LOCAL DEV ONLY — spins up a local Postgres+pgvector.
│                                   # Railway never builds from this; there is no
│                                   # Dockerfile for the app itself, by design (see §8).
├── pyproject.toml
├── .env.example
└── README.md
```

Do not collapse `app/` into one file. Keep vision-model code isolated behind
the interfaces in `app/vision/`.

---

## 5. Core interfaces (must stay swappable)

```python
class FaceDetector:
    def detect(self, image) -> list: ...

class FaceEmbedder:
    def embed(self, face) -> list[float]: ...   # length 128, see §2

class LivenessDetector:
    def check(self, face) -> float: ...

class FaceMatcher:
    def match(self, embedding): ...
```

Business logic (`services/`) must depend only on these interfaces, never on
`onnxruntime` or a concrete model file directly. Swapping any of the three
initial models for a newer one should never require touching `api/` or
`services/` — only the matching concrete class in `app/vision/`.

---

## 6. Database schema (PostgreSQL + pgvector)

```sql
-- people
id, employee_code, name, email, active, created_at, updated_at

-- face_embeddings  (a person can have MULTIPLE embeddings: frontal, angled, with glasses, etc.)
id, person_id, embedding VECTOR(128), model_name, model_version, created_at
-- 128 = SFace output dimension (see §2). Do not change without a migration
-- that also re-embeds every existing face_embeddings row with the new model.

-- doors
id, name, location, active, created_at

-- attendance
id, person_id, door_id, event_type, event_at, attendance_date,
similarity, liveness_score, created_at

-- prevent duplicate attendance at the DB level, not just in application code
UNIQUE (person_id, attendance_date, event_type)
```

The `UNIQUE` constraint is mandatory. Application-level "if not exists, insert"
checks are a race condition under concurrent requests; the DB constraint is
the real guard.

---

## 7. API surface

```
POST   /api/v1/people
POST   /api/v1/people/{id}/enroll
POST   /api/v1/recognition/identify
POST   /api/v1/attendance/mark
GET    /api/v1/attendance
GET    /api/v1/attendance/{person_id}
GET    /api/v1/attendance/today          # for the dashboard
GET    /api/v1/attendance/absentees      # for the dashboard
GET    /api/v1/attendance/export         # CSV, for the dashboard
GET    /api/v1/health
```

`POST /api/v1/recognition/identify` and `POST /api/v1/attendance/mark` are
**separate operations** — identify first, then mark attendance with the
returned `person_id`. Keeping them separate makes both independently testable.

`identify` response shape:

```json
{ "matched": true, "person_id": 123, "similarity": 0.87, "liveness": 0.94 }
```

or `{ "matched": false }` for unknown faces.

The dashboard-facing endpoints (`/today`, `/absentees`, `/export`) exist so the
Node.js dashboard never needs direct database credentials — it is a pure REST
client of `app/`.

---

## 8. Deploying to Railway — Railpack only, no Dockerfile

**Hard rule: this repo must never contain a `Dockerfile`.** Railway always
prefers a Dockerfile over Railpack if one is present in a service's root
directory, which would silently break the intended build path. If you ever
see a `Dockerfile` show up in `app/`, `dashboard/`, or the repo root, delete
it — it isn't supposed to be there.

### 8.1 Three things get provisioned in one Railway project

1. **Postgres, from Railway's pgvector template** — deploy Postgres via
   Railway's pgvector-enabled template (searchable in Railway's template
   gallery / "New" → "Database" → look for the pgvector template), not the
   plain Postgres template plus a manual `CREATE EXTENSION`. This gives you
   `pgvector` pre-installed. As a safety net, still include
   `CREATE EXTENSION IF NOT EXISTS vector;` as the first Alembic migration —
   it's a harmless no-op if the extension already exists, and protects you if
   the template ever changes.

2. **Backend service (`app/`)** — a Railway service whose **Root Directory**
   is the repo root (where `pyproject.toml` lives). Builder: `RAILPACK`
   (Railway's default; set it explicitly in `railway.json` for clarity).
   Railpack auto-detects the Python project from `pyproject.toml` /
   `requirements.txt`, but FastAPI has no implicit "web process" convention
   the way e.g. Django does, so you must set an explicit start command.

   `railway.json` (repo root, applies to the backend service):

   ```json
   {
     "$schema": "https://railway.com/railway.schema.json",
     "build": {
       "builder": "RAILPACK"
     },
     "deploy": {
       "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
       "healthcheckPath": "/api/v1/health",
       "healthcheckTimeout": 100,
       "restartPolicyType": "ON_FAILURE"
     }
   }
   ```

   Run Alembic migrations on boot (e.g. `alembic upgrade head &&` prefixed
   onto the start command, or a small entrypoint script referenced by
   `startCommand`) so schema changes ship automatically on deploy — there is
   no Dockerfile `ENTRYPOINT` to do this in, so it has to live in the start
   command itself or a plain shell script committed to the repo.

3. **Dashboard service (`dashboard/`)** — a separate Railway service whose
   **Root Directory** is `/dashboard`. Railpack auto-detects it as a Node
   project from `dashboard/package.json` and runs the standard
   install → build → start sequence (`npm ci`, then whatever `build`/`start`
   scripts are defined in `package.json`) with no config file required. Set
   `BACKEND_URL` for this service so it knows where to call the backend —
   prefer Railway's private networking (an internal hostname for the backend
   service within the same project) over the backend's public domain, since
   the dashboard never needs to be reachable from outside the backend's own
   trust boundary; check the current Railway dashboard/docs for the exact
   private-domain reference variable name for the backend service (it is
   typically exposed as a `RAILWAY_PRIVATE_DOMAIN`-style variable you can
   reference from another service in the same project).

### 8.2 System libraries for OpenCV / ONNX Runtime, without a Dockerfile

Because there's no Dockerfile, you can't add an `apt-get install` line
yourself. If `opencv-python-headless` or `onnxruntime` fail at import/runtime
with a missing shared library (this shows up as an `ImportError` mentioning a
`.so` file, most commonly `libgl1` or `libglib2.0-0` for OpenCV), set Railpack
service variables on the **backend** service rather than editing
`railway.json`'s `aptPackages` (that field has been unreliable in practice):

```
RAILPACK_DEPLOY_APT_PACKAGES=libgl1 libglib2.0-0
```

Use `RAILPACK_BUILD_APT_PACKAGES` instead if the missing library is only
needed at build/compile time. Always reach for `opencv-python-headless` (never
plain `opencv-python`) first — it avoids most of these dependencies to begin
with.

### 8.3 Environment variables (set in Railway's dashboard, never committed)

| Variable | Service | Notes |
|---|---|---|
| `DATABASE_URL` | backend | reference variable to the pgvector Postgres service |
| `RECOGNITION_THRESHOLD` | backend | calibrated per deployment, see §11 below |
| `LIVENESS_THRESHOLD` | backend | calibrated per deployment |
| `ATTENDANCE_TIMEZONE` | backend | e.g. `Asia/Kolkata` |
| `SECRET_KEY` / auth settings | backend | never commit real values |
| `CORS_ORIGINS` | backend | must include the dashboard's origin; no wildcards in production |
| `MODEL_DIR` | backend | defaults to `app/vision/models` |
| `BACKEND_URL` | dashboard | backend's private or public Railway domain |

Keep `.env.example` files (one at repo root for `app/`, one in `dashboard/`)
with placeholder values only.

### 8.4 Scaling note

Keep the backend at `numReplicas: 1` unless the temporal-confirmation state
(the recent-matches buffer described in §11) is moved out of in-process memory
into a shared store. With multiple backend replicas behind a load balancer,
"3 consecutive matches" tracked in a Python list on one instance won't see
frames routed to a different instance. The dashboard service can scale freely
since it's stateless and read-only.

### 8.5 What is *not* deployed here

The `door/` OpenCV camera client stays out of Railway entirely — it runs
wherever the physical camera is, and calls the Railway-hosted
`/api/v1/recognition/identify` and `/api/v1/attendance/mark` endpoints over
HTTPS using the backend service's Railway domain.

---

## 9. Enrollment

Enrollment happens before anyone can be recognized. Accept **multiple** images
per person (5–10), not one selfie. For each image: detect exactly one face
(YuNet), run a basic quality check (face large enough, not badly occluded,
single face), align using the detector's landmarks, embed (SFace), store.
Reject enrollment images that fail these checks rather than silently accepting
bad data.

---

## 10. Recognition pipeline (conceptual)

```python
def recognize(face_image):
    face = detector.detect(face_image)          # YuNetFaceDetector
    if face is None:
        return None
    live_score = liveness.check(face)            # MiniFASNetLivenessDetector
    if live_score < LIVENESS_THRESHOLD:
        return None
    embedding = embedder.embed(face)              # SFaceEmbedder, 128-d
    candidate = matcher.find_best_match(embedding)
    if candidate is None or candidate.similarity < RECOGNITION_THRESHOLD:
        return None
    return RecognitionResult(candidate.person_id, candidate.similarity, live_score)
```

Do not process every camera frame — sample at a low rate (e.g. ~5 fps out of
30 fps) on the `door/` client side.

---

## 11. Temporal confirmation & duplicate policy

- **Never mark attendance from a single frame.** Require the same person to
  be recognized consistently across several recent frames within a short
  window (e.g. 3 matches within 1 second) before confirming identity.
- **One check-in per person per local calendar day** by default (configurable:
  optional check-out, configurable cooldown). Enforce via the DB `UNIQUE`
  constraint in §6, not just application logic.
- Store all timestamps in UTC; compute `attendance_date` using a configured
  local timezone, not server-local time.

---

## 12. Thresholds — never hardcode

`RECOGNITION_THRESHOLD` and `LIVENESS_THRESHOLD` are environment-driven
config, never a magic number buried in business logic. They must be
calibrated per deployment (camera, lighting, distance) using real validation
data — genuine attempts vs. impostor attempts — not copied from a blog post.
Prefer also checking the similarity gap between the top two candidates, not
just the top score alone.

---

## 13. Security & privacy (this is biometric data)

- Never log raw face images or embeddings — in either `app/` or `dashboard/`.
- Require authentication/RBAC on all admin endpoints (`/people`, enrollment)
  and on the dashboard's UI itself.
- Never expose the recognition endpoint or the database publicly without auth.
- The dashboard talks to the backend only over its REST API (§7) — it must
  never be given `DATABASE_URL` or direct Postgres credentials.
- Use HTTPS everywhere (Railway provides this on generated domains by
  default).
- Define a retention/deletion policy for face images and embeddings.
- Add audit logging that records *who did what* without storing biometric
  payloads in the log itself.

---

## 14. Testing

`pytest` coverage (backend) should include: person creation, enrollment (valid
and invalid), unknown-face recognition, attendance insertion, **concurrent**
duplicate-attendance attempts (to prove the DB constraint holds), threshold
configuration, and timezone/date-boundary handling for `attendance_date`.
Vision-model tests should use small fixture images and assert output
**shapes/types** (e.g. a 128-length embedding, a bounding box tuple) rather
than exact recognition outcomes, to avoid flaky tests tied to specific model
weights.

The dashboard should have its own lightweight tests (e.g. via `node --test` or
a small test runner already in the Node ecosystem) covering `apiClient.js`
against a mocked backend response — it should never need a live Postgres
connection to test.

---

## 15. Local development

```bash
cp .env.example .env
docker compose up -d db          # local Postgres+pgvector for dev ONLY — not part of the Railway deploy path
alembic upgrade head
uvicorn app.main:app --reload
```

```bash
cd dashboard
cp .env.example .env             # set BACKEND_URL=http://localhost:8000
npm install
npm run dev
```

Run the door client separately against the local API:

```bash
python door/main.py --api-url http://localhost:8000
```

---

## 16. Build order (do not build everything at once)

1. Camera → detect (YuNet) → embed (SFace) → compare against a handful of
   enrolled people → print the matched name. No FastAPI, no DB yet. Get this
   reliable first.
2. Add enrollment (admin creates people, captures images, stores embeddings
   in Postgres/pgvector).
3. Add the attendance service and DB-level duplicate constraint.
4. Add temporal confirmation (require N consistent frames before confirming).
5. Add liveness (MiniFASNetV2).
6. Add the Node.js dashboard (today's attendance, absentees, search, CSV
   export) against the backend's REST API.
7. Wire up Railway deployment for `app/` and `dashboard/` as described in §8.

Only move to the next phase once the current one is reliable with real
camera data — most of the risk here is in the recognition pipeline, not in
FastAPI, Postgres, or the dashboard.

---

## 17. Conventions for whoever (human or Claude) works on this repo

- **Never add a Dockerfile.** This project deploys via Railway's Railpack
  builder for both services; a Dockerfile anywhere in a service's root
  directory silently overrides that.
- New vision models are added by implementing the existing interface
  (`FaceDetector`, `FaceEmbedder`, `LivenessDetector`, `FaceMatcher`) in
  `app/vision/`, never by changing `services/` or `api/` call sites.
- All three inference models run through `onnxruntime.InferenceSession` — do
  not introduce a second inference backend (e.g. OpenCV's `cv2.dnn` face
  wrappers) for consistency's sake, even when OpenCV has a built-in shortcut
  for a given model.
- Any new threshold, timeout, or policy constant goes into `core/config.py`
  (backend) as an environment-driven setting — never inline in business logic.
- Any change touching `attendance` must have a corresponding test for the
  concurrent-duplicate case.
- The dashboard must only talk to the backend's REST API — never add a direct
  Postgres client to `dashboard/`.
- Do not add code that logs a face image, an embedding vector, or any raw
  biometric payload, even at debug level.
