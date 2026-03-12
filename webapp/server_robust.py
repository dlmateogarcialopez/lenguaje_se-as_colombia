import http.server
import socketserver
import os

PORT = 8001
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHandler(http.server.SimpleHTTPRequestHandler):
    server_version = "RobustLSC/1.0"
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_GET(self):
        print(f"GET Request: {self.path}")
        if self.path.endswith('.js'):
            # Manual serving for JS to ensure MIME type
            local_path = self.path.lstrip('/')
            full_path = os.path.join(DIRECTORY, local_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                self.send_response(200)
                self.send_header('Content-type', 'application/javascript')
                self.send_header('Access-Control-Allow-Origin', '*')
                with open(full_path, 'rb') as f:
                    content = f.read()
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                return
        return super().do_GET()

print(f"Server LSC RobustLSC/1.0 iniciado en puerto {PORT}")
os.chdir(DIRECTORY)

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
