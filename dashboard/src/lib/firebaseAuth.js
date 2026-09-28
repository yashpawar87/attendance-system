import { cert, getApps, initializeApp } from 'firebase-admin/app';
import { getAuth } from 'firebase-admin/auth';

export const FIREBASE_SESSION_COOKIE = 'firebase_session';
export const FIREBASE_SESSION_SECONDS = 60 * 60 * 24 * 5;

export function firebaseAuthEnabled() {
  return process.env.FIREBASE_AUTH_ENABLED === 'true';
}

export function firebaseClientConfig() {
  return {
    apiKey: process.env.FIREBASE_API_KEY || '',
    authDomain: process.env.FIREBASE_AUTH_DOMAIN || '',
    projectId: process.env.FIREBASE_PROJECT_ID || '',
    appId: process.env.FIREBASE_APP_ID || '',
  };
}

function adminAuth() {
  const projectId = process.env.FIREBASE_PROJECT_ID;
  const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
  const privateKey = process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n');
  if (!projectId || !clientEmail || !privateKey) {
    throw new Error('Firebase Admin credentials are not configured.');
  }
  if (!getApps().length) {
    initializeApp({ credential: cert({ projectId, clientEmail, privateKey }) });
  }
  return getAuth();
}

function readCookie(request, name) {
  const header = request.get('cookie') || '';
  const pair = header.split(';').map((item) => item.trim()).find((item) => item.startsWith(`${name}=`));
  return pair ? decodeURIComponent(pair.slice(name.length + 1)) : '';
}

export function sessionCookie(request) {
  return readCookie(request, FIREBASE_SESSION_COOKIE);
}

export async function createFirebaseSession(idToken) {
  return adminAuth().createSessionCookie(idToken, { expiresIn: FIREBASE_SESSION_SECONDS * 1000 });
}

export async function verifyFirebaseSession(cookie) {
  return adminAuth().verifySessionCookie(cookie, true);
}

export function setFirebaseSession(response, cookie) {
  const secure = process.env.NODE_ENV === 'production' ? '; Secure' : '';
  response.setHeader('Set-Cookie', `${FIREBASE_SESSION_COOKIE}=${encodeURIComponent(cookie)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${FIREBASE_SESSION_SECONDS}${secure}`);
}

export function clearFirebaseSession(response) {
  response.setHeader('Set-Cookie', `${FIREBASE_SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`);
}

export async function requireFirebaseSession(request, response, next) {
  if (!firebaseAuthEnabled() || request.path.startsWith('/auth')) {
    next();
    return;
  }
  const cookie = sessionCookie(request);
  if (!cookie) {
    response.redirect(`/auth/login?returnTo=${encodeURIComponent(request.originalUrl || '/')}`);
    return;
  }
  try {
    request.firebaseUser = await verifyFirebaseSession(cookie);
    next();
  } catch (error) {
    clearFirebaseSession(response);
    response.redirect(`/auth/login?returnTo=${encodeURIComponent(request.originalUrl || '/')}`);
  }
}
