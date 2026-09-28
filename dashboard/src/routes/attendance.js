import { Router } from 'express';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { Readable } from 'node:stream';
import { getAbsentees, getAttendance, getCameraMetrics, getPeople, getToday } from '../lib/apiClient.js';
import multer from 'multer';
import { renderAnalytics, renderAttendance, renderDashboard, renderEmployeeForm, renderPeople, renderSettings } from '../views/dashboard.js';

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 8 * 1024 * 1024, files: 20 } });
let demoProcess = null;
let demoState = { running: false, startedAt: null, lastOutput: '', exitCode: null };
let cameraProcess = null;
let cameraState = { running: false, startedAt: null, lastOutput: '', exitCode: null };

function backendBaseUrl() {
  return (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
}

function backendHeaders() {
  return process.env.API_TOKEN ? { Authorization: `Bearer ${process.env.API_TOKEN}` } : {};
}

router.get('/camera/feed', async (req, res) => {
  const backendUrl = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
  const token = process.env.API_TOKEN || '';
  const sourceId = String(req.query.source || process.env.CAMERA_SOURCE_ID || 'door-camera');
  const upstream = await fetch(`${backendUrl}/api/v1/camera/feed?source_id=${encodeURIComponent(sourceId)}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!upstream.ok || !upstream.body) {
    res.status(upstream.status || 502).send('Camera feed unavailable');
    return;
  }
  res.status(200);
  res.setHeader('Content-Type', upstream.headers.get('content-type') || 'multipart/x-mixed-replace; boundary=frame');
  res.setHeader('Cache-Control', 'no-store');
  Readable.fromWeb(upstream.body).pipe(res);
});

router.get('/camera/metrics', async (req, res) => {
  const sourceId = String(req.query.source || process.env.CAMERA_SOURCE_ID || 'door-camera');
  try {
    const upstream = await fetch(`${backendBaseUrl()}/api/v1/camera/status?source_id=${encodeURIComponent(sourceId)}`, {
      headers: backendHeaders(),
    });
    const body = await upstream.text();
    res.status(upstream.status).type('application/json').send(body);
  } catch (error) {
    res.status(502).json({ detail: `Camera metrics unavailable: ${error.message}` });
  }
});

router.get('/demo/video', (req, res) => {
  const projectRoot = process.env.PROJECT_ROOT || path.resolve(process.cwd(), '..');
  const videoPath = process.env.DEMO_VIDEO || path.join(projectRoot, 'create_syn_data', 'P1E_S1_C1', 'chokepoint_P1E_S1_C1.mp4');
  res.sendFile(videoPath, { headers: { 'Cache-Control': 'no-store' } }, (error) => {
    if (error && !res.headersSent) res.status(error.statusCode || 404).send('Demo video unavailable');
  });
});

async function processBrowserFrame(req, res, replayDemo = false) {
  const file = req.file;
  if (!file) {
    res.status(400).json({ detail: 'A camera frame is required.' });
    return;
  }
  const sourceId = String(req.body.source_id || process.env.CAMERA_SOURCE_ID || 'door-camera');
  const recognitionForm = new FormData();
  recognitionForm.append('image', new Blob([file.buffer], { type: file.mimetype || 'image/jpeg' }), file.originalname || 'browser-frame.jpg');
  recognitionForm.append('source_id', sourceId);
  const recognitionPath = replayDemo ? '/api/v1/recognition/demo-identify' : '/api/v1/recognition/identify';
  const recognitionResponse = await fetch(`${backendBaseUrl()}${recognitionPath}`, {
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
  let attendance = null;
  if (result.matched) {
    const attendanceResponse = await fetch(`${backendBaseUrl()}/api/v1/attendance/${replayDemo ? 'demo-mark' : 'mark'}`, {
      method: 'POST',
      headers: { ...backendHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({
        person_id: result.person_id,
        similarity: result.similarity,
        liveness_score: result.liveness,
      }),
    });
    const attendanceText = await attendanceResponse.text();
    if (!attendanceResponse.ok) {
      res.status(attendanceResponse.status).type('application/json').send(attendanceText);
      return;
    }
    attendance = JSON.parse(attendanceText);
  }
  res.json({ ...result, attendance_marked: Boolean(attendance?.marked), attendance });
}

router.post('/browser/identify', upload.single('image'), async (req, res) => {
  try {
    await processBrowserFrame(req, res);
  } catch (error) {
    res.status(502).json({ detail: `Browser recognition unavailable: ${error.message}` });
  }
});

router.post('/browser/demo-identify', upload.single('image'), async (req, res) => {
  try {
    await processBrowserFrame(req, res, true);
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

router.get('/demo/status', (req, res) => {
  res.json({ ...demoState, pid: demoProcess?.pid || null });
});

router.post('/demo/start', (req, res) => {
  if (process.env.LOCAL_CONTROL_ENABLED !== 'true') {
    res.status(403).json({ detail: 'Local demo controls are disabled for this dashboard deployment.' });
    return;
  }
  if (cameraProcess && cameraState.running) {
    res.status(409).json({ detail: 'Stop the physical camera before starting the video demo.' });
    return;
  }
  if (demoProcess && demoState.running) {
    res.json({ ...demoState, pid: demoProcess.pid });
    return;
  }
  const projectRoot = process.env.PROJECT_ROOT || path.resolve(process.cwd(), '..');
  const pythonPath = process.env.PYTHON_BIN || path.join(projectRoot, '.venv', 'bin', 'python');
  const videoPath = process.env.DEMO_VIDEO || path.join(projectRoot, 'create_syn_data', 'P1E_S1_C1', 'chokepoint_P1E_S1_C1.mp4');
  demoState = { running: true, startedAt: new Date().toISOString(), lastOutput: '', exitCode: null };
  demoProcess = spawn(pythonPath, [
    'door/main.py', '--video', videoPath, '--api-url', backendBaseUrl(), '--token', process.env.API_TOKEN || 'change-me',
    '--source-id', process.env.CAMERA_SOURCE_ID || 'door-camera', '--sample-fps', process.env.DEMO_SAMPLE_FPS || '5', '--demo-replay',
  ], { cwd: projectRoot, env: process.env, stdio: ['ignore', 'pipe', 'pipe'] });
  const collect = (chunk) => {
    demoState.lastOutput = `${demoState.lastOutput}${chunk.toString()}`.slice(-2000);
  };
  demoProcess.stdout.on('data', collect);
  demoProcess.stderr.on('data', collect);
  demoProcess.on('close', (code) => {
    demoState = { ...demoState, running: false, exitCode: code };
    demoProcess = null;
  });
  demoProcess.on('error', (error) => {
    demoState = { ...demoState, running: false, exitCode: -1, lastOutput: `${demoState.lastOutput}${error.message}` };
    demoProcess = null;
  });
  res.status(202).json({ ...demoState, pid: demoProcess.pid });
});

router.post('/demo/stop', (req, res) => {
  if (demoProcess) demoProcess.kill('SIGTERM');
  res.json({ stopped: true });
});

router.get('/camera/status', (req, res) => {
  res.json({ ...cameraState, pid: cameraProcess?.pid || null });
});

router.post('/camera/start', (req, res) => {
  if (process.env.LOCAL_CONTROL_ENABLED !== 'true') {
    res.status(403).json({ detail: 'Local camera controls are disabled for this dashboard deployment.' });
    return;
  }
  if (demoProcess && demoState.running) {
    res.status(409).json({ detail: 'Stop the video demo before opening the physical camera.' });
    return;
  }
  if (cameraProcess && cameraState.running) {
    res.json({ ...cameraState, pid: cameraProcess.pid });
    return;
  }
  const projectRoot = process.env.PROJECT_ROOT || path.resolve(process.cwd(), '..');
  const pythonPath = process.env.PYTHON_BIN || path.join(projectRoot, '.venv', 'bin', 'python');
  cameraState = { running: true, startedAt: new Date().toISOString(), lastOutput: '', exitCode: null };
  cameraProcess = spawn(pythonPath, [
    'door/main.py', '--api-url', backendBaseUrl(), '--token', process.env.API_TOKEN || 'change-me',
    '--camera-index', process.env.CAMERA_INDEX || '0', '--source-id', process.env.CAMERA_SOURCE_ID || 'door-camera',
    '--sample-fps', process.env.CAMERA_SAMPLE_FPS || '5',
  ], { cwd: projectRoot, env: process.env, stdio: ['ignore', 'pipe', 'pipe'] });
  const collect = (chunk) => {
    cameraState.lastOutput = `${cameraState.lastOutput}${chunk.toString()}`.slice(-2000);
  };
  cameraProcess.stdout.on('data', collect);
  cameraProcess.stderr.on('data', collect);
  cameraProcess.on('close', (code) => {
    cameraState = { ...cameraState, running: false, exitCode: code };
    cameraProcess = null;
  });
  cameraProcess.on('error', (error) => {
    cameraState = { ...cameraState, running: false, exitCode: -1, lastOutput: `${cameraState.lastOutput}${error.message}` };
    cameraProcess = null;
  });
  res.status(202).json({ ...cameraState, pid: cameraProcess.pid });
});

router.post('/camera/stop', (req, res) => {
  if (cameraProcess) cameraProcess.kill('SIGTERM');
  res.json({ stopped: true });
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
