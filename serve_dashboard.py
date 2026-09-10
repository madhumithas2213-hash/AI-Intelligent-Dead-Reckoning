"""
Lightweight dashboard server for SIH26168 Intelligent Dead Reckoning.
Runs on standard Python without third-party dependencies.
"""

import http.server
import socketserver
import os
import sys

PORT = int(os.environ.get("PORT", 8000))
DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml", "outputs", "fusion"))

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path in ("/", "/dashboard", "/dashboard/"):
            self.path = "/dashboard.html"
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        # Keep logs clean
        sys.stdout.write(f"[AI-IDR Server] {format % args}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    # Allow port reuse and handle concurrent requests
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), DashboardHandler) as httpd:
        print(f"AI-IDR Dashboard server running at: http://0.0.0.0:{PORT}/")
        sys.stdout.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

