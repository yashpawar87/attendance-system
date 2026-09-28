import express from 'express';
import attendanceRouter from './routes/attendance.js';
import authRouter from './routes/auth.js';
import { requireFirebaseSession } from './lib/firebaseAuth.js';

const app = express();
app.disable('x-powered-by');
const dashboardToken = process.env.DASHBOARD_ACCESS_TOKEN || '';
app.use((req, res, next) => {
  if (req.path === '/health' || req.path.startsWith('/auth')) return next();
  if (!dashboardToken) return next();
  const authorization = req.get('authorization') || '';
  const bearer = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
  const basic = authorization.startsWith('Basic ')
    ? Buffer.from(authorization.slice(6), 'base64').toString('utf8').split(':').slice(1).join(':')
    : '';
  if (bearer === dashboardToken || basic === dashboardToken) return next();
  res.set('WWW-Authenticate', 'Basic realm="Attendance dashboard"').status(401).send('Authentication required');
});
app.use(express.json({ limit: '32kb' }));
app.use(express.urlencoded({ extended: false }));
app.get('/health', (req, res) => res.json({ status: 'ok' }));
app.use('/auth', authRouter);
app.use(requireFirebaseSession);
app.use('/', attendanceRouter);

const port = Number(process.env.PORT || 3000);
app.listen(port, '0.0.0.0', () => console.log(`Attendance dashboard listening on ${port}`));

export { app };
