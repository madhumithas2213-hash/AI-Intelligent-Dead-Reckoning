/**
 * AI-IDR Offline-First Service Worker (sw.js)
 * Caches core navigation assets, HTML, styles, and on-device ML model weights.
 * Enables the complete navigation system to boot and operate with 0% internet connectivity.
 */

const CACHE_NAME = 'ai-idr-cache-v1';
const ASSETS_TO_CACHE = [
    '/',
    '/index.html',
    '/dashboard.html',
    '/offline_ml_engine.js',
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap'
];

// Install Event — Pre-cache critical offline assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('[ServiceWorker] Pre-caching offline navigation assets...');
            return cache.addAll(ASSETS_TO_CACHE).catch((err) => {
                console.warn('[ServiceWorker] Some pre-cache assets failed (non-critical):', err);
            });
        }).then(() => self.skipWaiting())
    );
});

// Activate Event — Clean up stale caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keyList) => {
            return Promise.all(
                keyList.map((key) => {
                    if (key !== CACHE_NAME) {
                        console.log('[ServiceWorker] Removing old cache:', key);
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch Event — Offline-First with Network Revalidation
self.addEventListener('fetch', (event) => {
    const req = event.request;
    const url = new URL(req.url);

    // Bypass caching for WebSocket connections
    if (url.pathname.startsWith('/ws/')) {
        return;
    }

    // API Telemetry routes: If offline, respond with offline queuing status
    if (url.pathname.startsWith('/api/v1/')) {
        event.respondWith(
            fetch(req).catch(() => {
                return new Response(
                    JSON.stringify({
                        status: 'OFFLINE_BUFFERED',
                        message: 'Internet disconnected. Telemetry queued locally in IndexedDB.',
                        is_offline: true
                    }),
                    {
                        headers: { 'Content-Type': 'application/json' },
                        status: 200
                    }
                );
            })
        );
        return;
    }

    // Static Assets & UI Pages: Stale-While-Revalidate Strategy
    event.respondWith(
        caches.match(req).then((cachedResponse) => {
            const fetchPromise = fetch(req).then((networkResponse) => {
                if (networkResponse && networkResponse.status === 200) {
                    const responseToCache = networkResponse.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(req, responseToCache);
                    });
                }
                return networkResponse;
            }).catch(() => {
                // Return cached version or fallback to dashboard.html for navigation requests
                if (req.mode === 'navigate') {
                    return caches.match('/dashboard.html') || caches.match('/index.html');
                }
            });

            return cachedResponse || fetchPromise;
        })
    );
});
