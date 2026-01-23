#!/usr/bin/env python3
"""
Simple Python-based web applications as alternative to WAR files
These can be used if Java/Tomcat setup is problematic
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import sys

class App1Handler(BaseHTTPRequestHandler):
    """App 1: Hello World"""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = f"""
        <html><body>
        <h1>App 1 - Hello World</h1>
        <p>Request URI: {self.path}</p>
        <p>Query String: {self.command}</p>
        </body></html>
        """
        self.wfile.write(html.encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"status": "success", "app": "app1"}
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        # Log to stdout for access log simulation
        print(f"{self.client_address[0]} - - [{self.log_date_time_string()}] \"{args[0]}\" {args[1]} {args[2]}")

class App2Handler(BaseHTTPRequestHandler):
    """App 2: API Endpoint"""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"app": "app2", "path": self.path, "method": "GET"}
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"app": "app2", "method": "POST", "status": "received"}
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        print(f"{self.client_address[0]} - - [{self.log_date_time_string()}] \"{args[0]}\" {args[1]} {args[2]}")

class App3Handler(BaseHTTPRequestHandler):
    """App 3: Data Processing"""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"app": "app3", "endpoint": "data", "method": "GET"}
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"app": "app3", "endpoint": "data", "method": "POST"}
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        print(f"{self.client_address[0]} - - [{self.log_date_time_string()}] \"{args[0]}\" {args[1]} {args[2]}")

def run_server(port, handler_class):
    """Run a server on the specified port"""
    server = HTTPServer(('localhost', port), handler_class)
    print(f"Starting server on port {port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopping server on port {port}...")
        server.shutdown()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        app_num = int(sys.argv[1])
        port = 8080 + app_num - 1
        handlers = [App1Handler, App2Handler, App3Handler]
        run_server(port, handlers[app_num - 1])
    else:
        # Run all three apps in separate threads
        threads = []
        for i, handler in enumerate([App1Handler, App2Handler, App3Handler], 1):
            port = 8080 + i - 1
            thread = threading.Thread(target=run_server, args=(port, handler), daemon=True)
            thread.start()
            threads.append(thread)
            print(f"App {i} started on port {port}")
        
        print("\nAll applications running. Press Ctrl+C to stop.")
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            sys.exit(0)
