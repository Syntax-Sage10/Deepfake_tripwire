// Minimal service worker - mainly here to satisfy PWA installability
// requirements. It caches the app shell (this page + icons) so the UI can
// still load if you're briefly offline, but it deliberately does NOT try
// to cache or intercept /analyze*, since those need a live server and a
// live model to mean anything - going "offline-first" on those would just
// produce confusing stale/broken results.

const CACHE_NAME = "tripwire-shell-v1";
const SHELL_URLS = [
  "/",
  "/static/manifest.json",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_URLS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never intercept analysis requests - always go straight to the network.
  if (url.pathname.startsWith("/analyze")) {
    return;
  }

  // Only handle GET requests for the shell; let everything else pass through.
  if (event.request.method !== "GET") {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => {
      const network = fetch(event.request)
        .then((response) => {
          if (response && response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});