"""Tiny HTTP server that saves POST data to batch_temp.json"""
import http.server, json

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        data = self.rfile.read(length)
        with open('data/batch_temp.json', 'wb') as f:
            f.write(data)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'OK')
    def log_message(self, format, *args):
        pass  # silent

if __name__ == '__main__':
    server = http.server.HTTPServer(('127.0.0.1', 9876), Handler)
    print('Server ready on :9876')
    server.handle_request()  # handle one request then exit
