/* ZLAIservices Service Worker — PWA offline support + auto-update */
const CACHE_NAME = 'zlai-services-v3';
const ASSETS = [
    '/static/manifest.json',
    '/static/css/app.css',
    '/static/css/flatly.css',
    '/static/css/darkly.css',
    '/static/js/app.js',
    '/static/js/highlight.min.js',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
];

// Install: pre-cache core static assets (NOT root page — it's dynamic)
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(ASSETS.map(url => new Request(url, { credentials: 'include' })))
                .catch(err => console.log('[SW] Cache warm failed:', err));
        })
    );
    self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys => Promise.all(keys.map(k => caches.delete(k))))
    );
    self.clients.claim();
});

// Fetch: network-first for dynamic pages, cache-first for static assets
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // Don't cache API calls or POST requests
    if (url.pathname.startsWith('/send') || url.pathname.startsWith('/admin/') ||
        url.pathname.startsWith('/login') || url.pathname.startsWith('/create_account') ||
        event.request.method !== 'GET') {
        return; // Let network handle it
    }

    // Network-first for the root page (dynamic, session-based)
    if (url.pathname === '/' || url.pathname === '') {
        event.respondWith(
            fetch(event.request).then(response => {
                const clone = response.clone();
                caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                return response;
            }).catch(() => caches.match(event.request))
        );
        return;
    }

    // Cache-first for static assets (CSS, JS, icons)
    event.respondWith(
        caches.match(event.request).then(cached => {
            const fetchPromise = fetch(event.request).then(response => {
                if (response && response.status === 200) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                }
                return response;
            });
            return cached || fetchPromise;
        })
    );
});

// Notify all clients when a new version is installed
self.addEventListener('message', event => {
    if (event.data === 'skipWaiting') self.skipWaiting();
});

// Broadcast update to clients
self.addEventListener('controllerchange', () => {
    self.clients.matchAll().then(clients => {
        clients.forEach(client => client.postMessage({ type: 'SW_UPDATED' }));
    });
});
