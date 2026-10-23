"""Tiny localhost wrapper around python -m http.server.

The dataset viewer (tools/viewer/index.html) and metrics reports (reports/*/report.html)
both load images via relative paths. file:// CORS blocks that — serving from the repo
root via this script lets both pages just work.
"""
import argparse
import http.server
import os
import socket
import socketserver
import webbrowser
from pathlib import Path


def find_free_port(start=8765, span=50):
    for p in range(start, start + span):
        with socket.socket() as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port in [{}, {})".format(start, start + span))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=None, help="Listen port; default: 8765 (next free if busy)")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Directory to serve (default: cwd)")
    ap.add_argument("--open", action="store_true", help="Open the dataset viewer in the default browser")
    args = ap.parse_args()

    port = args.port if args.port is not None else find_free_port()
    os.chdir(args.root)
    print("Serving {}  →  http://localhost:{}/".format(args.root.resolve(), port))
    print("  Dataset viewer: http://localhost:{}/tools/viewer/".format(port))
    print("  Reports:        http://localhost:{}/reports/".format(port))
    if args.open:
        webbrowser.open("http://localhost:{}/tools/viewer/".format(port))
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
