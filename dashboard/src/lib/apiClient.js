const backendUrl = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
const apiToken = process.env.API_TOKEN || '';

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (apiToken) headers.Authorization = `Bearer ${apiToken}`;
  const response = await fetch(`${backendUrl}${path}`, { ...options, headers });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Backend request failed (${response.status}): ${detail}`);
  }
  return response;
}

export async function getToday() {
  return (await request('/api/v1/attendance/today')).json();
}

export async function getAbsentees() {
  return (await request('/api/v1/attendance/absentees')).json();
}

export async function getPeople() {
  return (await request('/api/v1/people')).json();
}

export async function getCameraMetrics(sourceId = process.env.CAMERA_SOURCE_ID || 'door-camera') {
  return (await request(`/api/v1/camera/status?source_id=${encodeURIComponent(sourceId)}`)).json();
}

export async function getAttendance(params = {}) {
  const query = new URLSearchParams();
  if (params.attendance_date) query.set('attendance_date', params.attendance_date);
  if (params.person_id) query.set('person_id', params.person_id);
  const suffix = query.toString() ? `?${query}` : '';
  return (await request(`/api/v1/attendance${suffix}`)).json();
}

export async function getExportUrl(attendanceDate = '') {
  const suffix = attendanceDate ? `?attendance_date=${encodeURIComponent(attendanceDate)}` : '';
  return `${backendUrl}/api/v1/attendance/export${suffix}`;
}

export { request };
