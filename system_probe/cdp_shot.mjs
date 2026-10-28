// Read-only UI evidence capture via Chrome DevTools Protocol.
// - reuses an already-running headless Chrome on 127.0.0.1:9222 (Windows Chrome from WSL)
// - injects the existing admin session cookie (obtained by one POST /login)
// - navigates to read-only pages and saves PNG screenshots + visible text into the session workspace
import { writeFileSync, mkdirSync } from 'node:fs';

const CDP = 'http://127.0.0.1:9222';
const BASE = 'https://127.0.0.1:8443';
const COOKIE = process.env.SESSION_COOKIE || '';
const OUT = new URL('.', import.meta.url).pathname; // system_probe/
const SHOTS = OUT + 'screenshots/';
mkdirSync(SHOTS, { recursive: true });

const pages = [
  ['02_admin_projects', '/admin/projects', '项目管理'],
  ['03_admin_db_tables', '/admin/db_tables_overview', '数据表总览'],
  ['04_admin_users', '/admin/users', '用户管理'],
  ['05_wiki_index', '/wiki/index', '知识库 Wiki'],
  ['06_admin_analytics', '/admin/analytics', '运行分析'],
  ['07_cases', '/cases', '案例库'],
];

function wsSend(ws, id, method, params = {}) {
  return new Promise((resolve, reject) => {
    const onMsg = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id === id) { ws.removeEventListener('message', onMsg); resolve(m); }
    };
    ws.addEventListener('message', onMsg);
    ws.send(JSON.stringify({ id, method, params }));
    setTimeout(() => reject(new Error('timeout ' + method)), 45000);
  });
}

async function main() {
  const t = await (await fetch(CDP + '/json/new?about:blank', { method: 'PUT' })).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0;
  const send = (method, params) => wsSend(ws, ++id, method, params);

  await send('Network.enable');
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1100, deviceScaleFactor: 1, mobile: false });

  const ck = await send('Network.setCookie', {
    name: 'session', value: COOKIE, url: BASE + '/', path: '/',
    secure: true, httpOnly: true, sameSite: 'Lax',
  });
  console.log('setCookie:', JSON.stringify(ck.result));

  const results = [];
  for (const [name, path, label] of pages) {
    await send('Page.navigate', { url: BASE + path });
    await new Promise(r => setTimeout(r, 2500));
    const txt = await send('Runtime.evaluate', {
      expression: 'document.body ? document.body.innerText.slice(0,1500) : ""',
      returnByValue: true,
    });
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(SHOTS + name + '.png', Buffer.from(shot.result.data, 'base64'));
    const body = (txt.result?.result?.value || '').replace(/\n{2,}/g, '\n').trim();
    writeFileSync(OUT + 'evidence/page_' + name + '.txt',
      `URL: ${BASE + path}\nLABEL: ${label}\n\n--- visible text (first 1500 chars) ---\n${body}\n`);
    console.log(`[${name}] ${path} -> png ${shot.result.data.length}b64; text ${body.length} chars`);
  }
  await fetch(CDP + '/json/close/' + t.id);
  ws.close();
}
main().catch(e => { console.error('ERR', e.message); process.exit(1); });
