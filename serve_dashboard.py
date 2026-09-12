"""
Server entrypoint for SIH26168 Intelligent Dead Reckoning.
Launches the full FastAPI backend (with SQLite database, live smartphone telemetry,
AI model inference, and Dead Reckoning engine) on 0.0.0.0:8000.
Falls back to lightweight static server if uvicorn is not installed.
"""

import os
import sys

PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

def run_fastapi_server():
    import uvicorn
    print(f"================================================================")
    print(f">> AI-IDR Full-Stack Backend & Mobile Sensor Server Starting")
    print(f">> Local Access:      http://localhost:{PORT}/")
    print(f">> Smartphone Access: http://<Your-Computer-IP>:{PORT}/")
    print(f">> SQLite Database:   idr_telemetry.db (WAL Mode Active)")
    print(f">> AI Inference:      PyTorch GRU Velocity Regressor Attached")
    print(f"================================================================")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False, access_log=True)

def run_fallback_static_server():
    import http.server
    import socketserver
    dashboard_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml", "outputs", "fusion"))
    
    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=dashboard_dir, **kwargs)

        def do_GET(self):
            if self.path in ("/", "/dashboard", "/dashboard/"):
                self.path = "/dashboard.html"
            return super().do_GET()

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((HOST, PORT), DashboardHandler) as httpd:
        print(f"AI-IDR Dashboard static server running at: http://{HOST}:{PORT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    try:
        run_fastapi_server()
    except ImportError:
        print("[Warning] uvicorn not found. Falling back to lightweight static server.")
        run_fallback_static_server()
