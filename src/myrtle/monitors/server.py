import contextlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import requests
import socket
import sys
from threading import Thread
import time
from myrtle.config import monitor_host, monitor_port, js_dir, write_config_js

_pause = 1.0  # seconds
_protocol = 'HTTP/1.0'

global httpd

# Copy the host and port from config to a js-readable file  
write_config_js()


def serve():
    global httpd

    addr = (monitor_host, monitor_port)
    KillableHandler.protocol_version = _protocol
    httpd = MonitorWebServer(addr, KillableHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, exiting.")
        sys.exit(0)

    httpd.server_close()


def shutdown():

    def kill_server(host, port):
        n_retries = 3
        for _ in range(n_retries):
            try:
                requests.post(f"http://{host}:{port}/kill_server")
                return
            except requests.exceptions.ConnectionError:
                pass

    Thread(target=kill_server, args=(monitor_host, monitor_port)).start()

    # Give the server time to wind down before resuming activities.
    time.sleep(_pause)


class MonitorWebServer(ThreadingHTTPServer):
    # def server_bind(self):
        # This is a trick to make the socket instantly re-usable
        # in case of [Errno 98] Address already in use. from
        # https://stackoverflow.com/questions/6380057/address-already-in-use-error-when-binding-a-socket-in-python/18858817#18858817
        # self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # return super().server_bind()

    def finish_request(self, request, client_address):
        self.RequestHandlerClass(
            request,
            client_address,
            self,
            directory=js_dir,
        )

    def log_message(self, format, *args):
        pass


"""
Stolen and modified from
https://stackoverflow.com/questions/10085996/shutdown-socketserver-serve-forever-in-one-thread-python-application
"""
class KillableHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        global httpd
        if self.path.startswith('/kill_server'):
            print("Server is going down")
            def kill_me_please(server):
                server.shutdown()
            Thread(target=kill_me_please, args=(httpd,)).start()
            self.send_error(500)


if __name__ == "__main__":
    serve()
