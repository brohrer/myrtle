from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import requests
import sys
from threading import Thread
import time
from myrtle.config import monitor_host, monitor_port, js_dir, write_config_js

_short_pause = 0.01  # seconds
_pause = 1.0  # seconds
_protocol = "HTTP/1.0"

global httpd

# Copy the host and port from config to a js-readable file
write_config_js()


def serve():
    global httpd

    addr = (monitor_host, monitor_port)
    KillableHandler.protocol_version = _protocol
    httpd = MonitorWebServer(addr, KillableHandler)

    httpd.serve_forever()
    httpd.server_close()


def shutdown():
    def kill_server(host, port):
        n_retries = 3
        for _ in range(n_retries):
            try:
                try:
                    requests.post(f"http://{host}:{port}/kill_server")
                except requests.exceptions.ChunkedEncodingError:
                    # Catch the case where some communcations get interrrupted,
                    # but don't do anything about it.
                    pass
                return
            except requests.exceptions.ConnectionError:
                time.sleep(_short_pause)

        sys.exit(0)

    t_kill = Thread(target=kill_server, args=(monitor_host, monitor_port))
    t_kill.start()
    t_kill.join(_pause)


class MonitorWebServer(ThreadingHTTPServer):
    def finish_request(self, request, client_address):
        self.RequestHandlerClass(
            request,
            client_address,
            self,
            directory=js_dir,
        )

    def log_message(self, format, *args):
        pass


class KillableHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        global httpd
        if self.path.startswith("/kill_server"):
            httpd.shutdown()
            self.send_error(500)

    # This silences all server messages to stdout.
    # If you're debugging the server, comment this out.
    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    serve()
