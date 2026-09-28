import { Router } from 'express';
import { clearFirebaseSession, createFirebaseSession, firebaseAuthEnabled, firebaseClientConfig, setFirebaseSession } from '../lib/firebaseAuth.js';
import { renderLoginPage } from '../views/auth.js';

const router = Router();

router.get('/login', (req, res) => {
  if (!firebaseAuthEnabled()) {
    res.status(503).send(renderLoginPage({ firebaseConfig: {}, error: 'Firebase authentication is not enabled on this deployment.' }));
    return;
  }
  res.send(renderLoginPage({ firebaseConfig: firebaseClientConfig(), returnTo: req.query.returnTo }));
});

router.post('/session', async (req, res) => {
  if (!firebaseAuthEnabled()) {
    res.status(503).json({ detail: 'Firebase authentication is not enabled.' });
    return;
  }
  const authorization = req.get('authorization') || '';
  const idToken = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
  if (!idToken) {
    res.status(401).json({ detail: 'Firebase ID token is required.' });
    return;
  }
  try {
    const sessionCookie = await createFirebaseSession(idToken);
    setFirebaseSession(res, sessionCookie);
    res.json({ authenticated: true });
  } catch (error) {
    res.status(401).json({ detail: `Firebase sign-in failed: ${error.message}` });
  }
});

router.post('/logout', (req, res) => {
  clearFirebaseSession(res);
  res.redirect('/auth/login');
});

export default router;
