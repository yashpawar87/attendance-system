import { Router } from 'express';
import { Readable } from 'node:stream';
import { getAbsentees, getAttendance, getCameraMetrics, getPeople, getToday } from '../lib/apiClient.js';
import multer from 'multer';
import { attendanceTable, renderAnalytics, renderAttendance, renderDashboard, renderEmployeeForm, renderPeople, renderSettings } from '../views/dashboard.js';

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 8 * 1024 * 1024, files: 20 } });
function backendBaseUrl() {
  return (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
}

function backendHeaders() {
  return process.env.API_TOKEN ? { Authorization: `Bearer ${process.env.API_TOKEN}` } : {};
}

router.get('/live/overview', async (req, res) => {
  try {
    const [today, absentees, people, allAttendance, cameraStatus] = await Promise.all([
      getToday(),
      getAbsentees(),
      getPeople(),
      getAttendance(),
      getCameraMetrics().catch(() => ({ intruder_events: 0 })),
    ]);
    const presentToday = new Set(today.map((row) => row.person_id)).size;
    const totalWeek = new Set(allAttendance.map((row) => `${row.attendance_date || String(row.event_at).slice(0, 10)}-${row.person_id}`)).size;
    res.json({
      present_today: presentToday,
      attendance_rate: people.length ? (presentToday / people.length) * 100 : 0,
      needs_attention: absentees.length,
      total_week: totalWeek,
      intruder_events: Number(cameraStatus.intruder_events || 0),
    });
  } catch (error) {
    res.status(502).json({ detail: `Dashboard metrics unavailable: ${error.message}` });
  }
});

router.get('/live/activity', async (req, res) => {
  try {
    const [today, people] = await Promise.all([getToday(), getPeople()]);
    res.json({ html: attendanceTable(today, people, 6), total: today.length });
  } catch (error) {
    res.status(502).json({ detail: `Recent activity unavailable: ${error.message}` });
  }
});

router.get('/demo/video', (req, res) => {
  fetch(`${backendBaseUrl()}/api/v1/demo/video`, { headers: backendHeaders() })
    .then(async (upstream) => {
      if (!upstream.ok || !upstream.body) {
        res.status(upstream.status || 502).send(await upstream.text());
        return;
      }
      res.status(200);
      res.setHeader('Content-Type', upstream.headers.get('content-type') || 'video/mp4');
      res.setHeader('Cache-Control', 'no-store');
      const length = upstream.headers.get('content-length');
      if (length) res.setHeader('Content-Length', length);
      Readable.fromWeb(upstream.body).pipe(res);
    })
    .catch((error) => res.status(502).send(`Demo video unavailable: ${error.message}`));
});

router.get('/camera/feed', (req, res) => {
  const sourceId = req.query.source || 'demo-replay';
  fetch(`${backendBaseUrl()}/api/v1/camera/feed?source_id=${encodeURIComponent(sourceId)}`, { headers: backendHeaders() })
    .then(async (upstream) => {
      if (!upstream.ok || !upstream.body) {
        res.status(upstream.status || 502).send(await upstream.text());
        return;
      }
      res.status(200);
      res.setHeader('Content-Type', upstream.headers.get('content-type') || 'multipart/x-mixed-replace; boundary=frame');
      res.setHeader('Cache-Control', 'no-store');
      Readable.fromWeb(upstream.body).pipe(res);
    })
    .catch((error) => res.status(502).send(`Camera feed unavailable: ${error.message}`));
});

async function processDemoFrame(req, res) {
  const file = req.file;
  if (!file) {
    res.status(400).json({ detail: 'A demo video frame is required.' });
    return;
  }
  const sourceId = String(req.body.source_id || process.env.CAMERA_SOURCE_ID || 'door-camera');
  const recognitionForm = new FormData();
  recognitionForm.append('image', new Blob([file.buffer], { type: file.mimetype || 'image/jpeg' }), file.originalname || 'demo-frame.jpg');
  recognitionForm.append('source_id', sourceId);
  const recognitionResponse = await fetch(`${backendBaseUrl()}/api/v1/recognition/demo-identify`, {
    method: 'POST',
    headers: backendHeaders(),
    body: recognitionForm,
  });
  const recognitionText = await recognitionResponse.text();
  if (!recognitionResponse.ok) {
    res.status(recognitionResponse.status).type('application/json').send(recognitionText);
    return;
  }
  const result = JSON.parse(recognitionText);
  const matches = Array.isArray(result.matches) && result.matches.length
    ? result.matches
    : result.matched
      ? [{ person_id: result.person_id, similarity: result.similarity, liveness: result.liveness }]
      : [];
  const attendances = [];
  for (const match of matches) {
    const attendanceResponse = await fetch(`${backendBaseUrl()}/api/v1/attendance/demo-mark`, {
      method: 'POST',
      headers: { ...backendHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({
        person_id: match.person_id,
        similarity: match.similarity,
        liveness_score: match.liveness,
      }),
    });
    const attendanceText = await attendanceResponse.text();
    if (!attendanceResponse.ok) {
      res.status(attendanceResponse.status).type('application/json').send(attendanceText);
      return;
    }
    attendances.push(JSON.parse(attendanceText));
  }
  res.json({
    ...result,
    attendance_marked: attendances.some((item) => item.marked),
    attendance: attendances[0] || null,
    attendances,
  });
}

router.post('/browser/demo-identify', upload.single('image'), async (req, res) => {
  try {
    await processDemoFrame(req, res);
  } catch (error) {
    res.status(502).json({ detail: `Browser demo unavailable: ${error.message}` });
  }
});

router.get('/people/:personId/photo', async (req, res) => {
  const backendUrl = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
  const token = process.env.API_TOKEN || '';
  const upstream = await fetch(`${backendUrl}/api/v1/people/${encodeURIComponent(req.params.personId)}/photo`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!upstream.ok || !upstream.body) {
    res.status(upstream.status || 404).send('Photo unavailable');
    return;
  }
  res.status(200);
  res.setHeader('Content-Type', upstream.headers.get('content-type') || 'image/jpeg');
  res.setHeader('Cache-Control', 'public, max-age=300');
  Readable.fromWeb(upstream.body).pipe(res);
});

router.get('/', async (req, res) => {
  try {
    const [today, absentees, people, allAttendance, cameraStatus] = await Promise.all([
      getToday(), getAbsentees(), getPeople(), getAttendance(), getCameraMetrics().catch(() => ({ intruder_events: 0, unknown_face_detections: 0 })),
    ]);
    res.send(renderDashboard({ today, absentees, people, allAttendance, cameraStatus }));
  } catch (error) {
    res.status(502).send(renderDashboard({ today: [], absentees: [], people: [], allAttendance: [], cameraStatus: {}, error: error.message }));
  }
});

router.get('/people', async (req, res) => {
  try {
    const [people, today] = await Promise.all([getPeople(), getToday()]);
    res.send(renderPeople({ people, today, notice: req.query.created ? 'Employee created and face samples enrolled.' : '' }));
  } catch (error) {
    res.status(502).send(renderPeople({ people: [], today: [], error: error.message }));
  }
});

router.get('/people/new', (req, res) => {
  res.send(renderEmployeeForm());
});

router.post('/people/create', upload.fields([
  { name: 'photo', maxCount: 1 },
  { name: 'enrollment_images', maxCount: 20 },
]), async (req, res) => {
  try {
    const response = await fetch(`${backendBaseUrl()}/api/v1/people`, {
      method: 'POST',
      headers: { ...backendHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ employee_code: req.body.employee_code, name: req.body.name, email: req.body.email || null }),
    });
    if (!response.ok) throw new Error(await response.text());
    const person = await response.json();
    const files = req.files?.enrollment_images || [];
    const enrollment = new FormData();
    files.forEach((file) => enrollment.append('images', new Blob([file.buffer], { type: file.mimetype }), file.originalname));
    const enrollmentResponse = await fetch(`${backendBaseUrl()}/api/v1/people/${person.id}/enroll`, {
      method: 'POST',
      headers: backendHeaders(),
      body: enrollment,
    });
    if (!enrollmentResponse.ok) throw new Error(await enrollmentResponse.text());
    const photo = req.files?.photo?.[0];
    if (photo) {
      const photoForm = new FormData();
      photoForm.append('image', new Blob([photo.buffer], { type: photo.mimetype }), photo.originalname);
      const photoResponse = await fetch(`${backendBaseUrl()}/api/v1/people/${person.id}/photo`, {
        method: 'POST',
        headers: backendHeaders(),
        body: photoForm,
      });
      if (!photoResponse.ok) throw new Error(await photoResponse.text());
    }
    res.redirect('/people?created=1');
  } catch (error) {
    res.status(400).send(renderEmployeeForm({ error: `Could not create employee: ${error.message}` }));
  }
});

router.get('/attendance', async (req, res) => {
  try {
    const [rows, people] = await Promise.all([getAttendance(), getPeople()]);
    res.send(renderAttendance({ rows, people }));
  } catch (error) {
    res.status(502).send(renderAttendance({ rows: [], people: [], error: error.message }));
  }
});

router.get('/analytics', async (req, res) => {
  try {
    const [today, people, allAttendance] = await Promise.all([getToday(), getPeople(), getAttendance()]);
    res.send(renderAnalytics({ today, people, allAttendance }));
  } catch (error) {
    res.status(502).send(renderAnalytics({ today: [], people: [], allAttendance: [], error: error.message }));
  }
});

router.get('/settings', (req, res) => {
  res.send(renderSettings());
});

router.get('/search', async (req, res) => {
  try {
    const [rows, people] = await Promise.all([
      getAttendance({ person_id: req.query.person_id, attendance_date: req.query.date }),
      getPeople(),
    ]);
    res.send(renderAttendance({ rows, people }));
  } catch (error) {
    res.status(502).send(renderAttendance({ rows: [], people: [], error: error.message }));
  }
});

router.get('/export', async (req, res) => {
  const backendUrl = backendBaseUrl();
  const token = process.env.API_TOKEN || '';
  const suffix = req.query.date ? `?attendance_date=${encodeURIComponent(req.query.date)}` : '';
  const upstream = await fetch(`${backendUrl}/api/v1/attendance/export${suffix}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!upstream.ok || !upstream.body) {
    res.status(upstream.status || 502).send('Export unavailable');
    return;
  }
  res.status(200);
  res.setHeader('Content-Type', upstream.headers.get('content-type') || 'text/csv');
  res.setHeader('Content-Disposition', upstream.headers.get('content-disposition') || 'attachment; filename=attendance.csv');
  Readable.fromWeb(upstream.body).pipe(res);
});

export default router;
