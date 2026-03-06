import http.server
import socketserver
import mimetypes

PORT = 8001

# Force the mimetypes registry before starting the server
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')
mimetypes.add_type('image/svg+xml', '.svg')

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Additional manual override for safety
        if str(self.path).endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        super().end_headers()

# To allow reusing the port immediately
socketserver.TCPServer.allow_reuse_address = True

try:
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Serving at port {PORT} with enforced JavaScript MIME types...")
        httpd.serve_forever()
except OSError as e:
    print(f"Error binding to port {PORT}: {e}")
