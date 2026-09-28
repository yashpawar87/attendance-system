import test from 'node:test';
import assert from 'node:assert/strict';
import { getToday } from './apiClient.js';

test('api client parses a successful backend response', async () => {
  const originalFetch = global.fetch;
  global.fetch = async (url) => {
    assert.equal(url, 'http://localhost:8000/api/v1/attendance/today');
    return { ok: true, json: async () => [{ id: 1 }] };
  };
  try {
    assert.deepEqual(await getToday(), [{ id: 1 }]);
  } finally {
    global.fetch = originalFetch;
  }
});
