#!/usr/bin/env python3
"""Stand-in for the MineLLM server (LAMM/src/mllm_api_mc_v1.5.py) for CPU rehearsals.

Same interface as the real server:
  POST /image, multipart form: file (image), text (question), is_del (int)
  -> 200 {"result": 1, "answer": "<fixed answer>"}
Like the real server: missing file -> 400, non-integer is_del -> 500, GET -> 405/404.

Standard library only. Each request is logged as one JSON line on stdout
(question, image size, is_del) so a rehearsal can check what the agent sent.
With --save-dir, received images are kept there for inspection.

  python3 tools/fake_minellm_server.py [--port 25541] [--answer TEXT] [--save-dir DIR]
"""
import argparse
import json
import os
import sys
import time
import uuid
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_ANSWER = "There is a tree in front of you. (FAKE MineLLM answer)"


def parse_multipart(content_type, body):
    """Return {field_name: (filename_or_None, bytes)} for a multipart/form-data body."""
    msg = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    fields = {}
    if not msg.is_multipart():
        return fields
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = (part.get_filename(), part.get_payload(decode=True) or b"")
    return fields


class Handler(BaseHTTPRequestHandler):
    server_version = "FakeMineLLM/1.0"

    def _reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # keep stdout for the JSON request log
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path == "/image":
            self._reply(405, {"error": "Method Not Allowed"})
        else:
            self._reply(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path != "/image":
            self._reply(404, {"error": "Not Found"})
            return
        t0 = time.time()
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        fields = parse_multipart(self.headers.get("Content-Type", ""), body)
        if "file" not in fields:
            self._reply(400, {"error": "No file part in the request"})
            return
        filename, image = fields["file"]
        if not filename:
            self._reply(400, {"error": "No selected file"})
            return
        text = fields.get("text", (None, b""))[1].decode("utf-8", "replace")
        is_del_raw = fields.get("is_del", (None, None))[1]
        try:
            is_del = None if is_del_raw is None else int(is_del_raw.decode())
        except ValueError:
            # The real server calls int(is_del) and crashes with HTTP 500.
            self._reply(500, {"error": "invalid is_del (real server would crash here)"})
            return
        saved = None
        if self.server.save_dir:
            saved = os.path.join(self.server.save_dir, f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg")
            with open(saved, "wb") as f:
                f.write(image)
        print(json.dumps({"ts": round(t0, 3), "question": text, "filename": filename,
                          "image_bytes": len(image), "is_del": is_del, "saved": saved}), flush=True)
        self._reply(200, {"result": 1, "answer": self.server.answer})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25541)
    ap.add_argument("--answer", default=DEFAULT_ANSWER)
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.answer, srv.save_dir = args.answer, args.save_dir
    sys.stderr.write(f"fake MineLLM listening on http://{args.host}:{args.port}/image\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()
