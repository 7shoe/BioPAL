"""A fake OpenAI-compatible server on a loopback port, standing in for vLLM: the model list, the version and one
chat completion per request. `mode` decides how it answers; `requests` lists what it was asked."""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REVISION = "0123456789abcdef0123456789abcdef01234567"


class Endpoint:
    def __init__(self, model="toy-llm", root=f"/weights/models--toy--llm/snapshots/{REVISION}", version="0.0-fake"):
        self.model, self.root, self.version, self.mode, self.requests = model, root, version, "ok", []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, body, headers=()):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(data))), *headers):
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                endpoint.requests.append(("GET", self.path, None))
                if self.path == "/v1/models":
                    self.send(200, dict(object="list", data=[dict(id=endpoint.model, root=endpoint.root,
                                                                  max_model_len=4096, owned_by="vllm")]))
                elif self.path == "/version" and endpoint.version:
                    self.send(200, dict(version=endpoint.version))
                else:
                    self.send(404, dict(error="not found"))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append(("POST", self.path, body))
                mode = endpoint.mode
                if self.path != "/v1/chat/completions":   # where a followed redirect would arrive
                    return self.send(200, endpoint.completion(json.dumps(endpoint.answer(body))))
                if mode == "sleep":
                    time.sleep(3)
                if mode == "http500":
                    return self.send(500, dict(error="the model fell over"))
                if mode == "redirect":                    # a 302 is what a stock client follows, as a GET
                    return self.send(302, b"", [("Location", endpoint.url.replace("/v1", "/elsewhere/v1")
                                                 + "/chat/completions")])
                answer = endpoint.answer(body)
                if mode == "keep":
                    answer.update(action="keep", target="", predicted_sign="no_change")
                if mode == "ordinary":
                    answer.update(action="ordinary_weighting", target="")
                if mode == "readout":
                    answer.update(action="set_readout", target="block_2")
                if mode == "junk":
                    answer = dict(action="dance")
                if mode == "off_menu":
                    answer["target"] = "no_such_cohort"
                if mode == "real_name":                   # the real name of a cohort: the masked expert never read it
                    answer["target"] = "gc:low"
                if mode == "echo":                        # a reply that repeats what the brief said in its note
                    said = re.search(r"TASK NOTE.*?\n(.*?)\nEND OF TASK NOTE", body["messages"][1]["content"], re.S)
                    answer.update(hypothesis=said.group(1) if said else "no note here", evidence_rows=["r01", "the note"])
                content = json.dumps(answer)
                if mode == "prose":
                    content = "I would keep the configuration."
                if mode == "think":
                    content = "<think>the first cohort looks bad</think>\n" + content
                if mode == "truncated":
                    content = content[:len(content) // 2]
                self.send(200, endpoint.completion(content))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def answer(body):
        """The first cohort on offer, as a full answer."""
        offered = re.search(r"set_cohort <cohort>.*On offer: (.*)", body["messages"][1]["content"]).group(1).split(", ")
        return dict(layer_evidence="block_2", evidence_rows=["r01"], action="set_cohort", target=offered[0],
                    predicted_sign="lower", hypothesis="the first cohort on offer")

    def completion(self, content):
        return dict(id="chatcmpl-1", object="chat.completion", model=self.model,
                    choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content=content))],
                    usage=dict(prompt_tokens=900, completion_tokens=60, total_tokens=960))

    def posts(self):
        return [body for method, path, body in self.requests if method == "POST"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
