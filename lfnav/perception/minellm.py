"""MineLLM template baseline. This is MP5 perception without the GPT-4 questions.

POST {url} as multipart form fields file, text, is_del. The JSON body has "answer".
Default URL is http://127.0.0.1:25541/image.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from lfnav.types import ObjectBelief, SceneState

TEMPLATES = {
    "present": "Is there a {name} in this image? Answer yes or no.",
    "biome": "Which biome is this? Answer one of: forest, plains, desert, savanna, taiga, jungle, other.",
    "time": "Is it day or night? Answer day or night.",
    "weather": "Is the weather clear or rain? Answer clear or rain.",
    "brightness": "Is the brightness sufficient or insufficient? Answer sufficient or insufficient.",
}


def parse_yes(text: str) -> float:
    word = (text or "").strip().lower()
    if word.startswith("yes"):
        return 1.0
    if word.startswith("no"):
        return 0.0
    if "yes" in word and "no" not in word:
        return 1.0
    return 0.0


def parse_choice(text: str, choices: list[str]) -> dict:
    word = (text or "").strip().lower()
    for choice in choices:
        if choice in word:
            return {name: 1.0 if name == choice else 0.0 for name in choices}
    return {name: 0.0 for name in choices}


def scene_from_answers(answers: dict, objects: list[str], labels: dict) -> SceneState:
    beliefs = {}
    for name in objects:
        present = float(answers.get(f"present:{name}", 0.0))
        beliefs[name] = ObjectBelief(
            present=present,
            direction={"left": 0.0, "center": present, "right": 0.0},
            distance={"near": 0.0, "mid": present, "far": 0.0},
        )
    return SceneState(
        objects=beliefs,
        biome=parse_choice(str(answers.get("biome", "")), list(labels.get("biomes") or [])),
        time=parse_choice(str(answers.get("time", "")), list(labels.get("time") or [])),
        weather=parse_choice(str(answers.get("weather", "")), list(labels.get("weather") or [])),
        brightness=parse_choice(str(answers.get("brightness", "")), list(labels.get("brightness") or [])),
    )


def post_image(url: str, image_path: str, text: str, timeout: float = 60.0) -> str:
    boundary = "lfnavboundary"
    file_bytes = Path(image_path).read_bytes()
    parts = []
    for field, value in (("text", text), ("is_del", "true")):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"frame.png\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode() + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    return str(payload.get("answer", ""))


class MineLLMPerception:
    name = "minellm"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = cfg.get("mllm_url", "http://127.0.0.1:25541/image")
        self.labels = cfg.get("labels") or {}
        self.objects = list(self.labels.get("objects") or [])

    def features(self, frame):
        return None

    def perceive(self, frame) -> SceneState:
        path = frame if isinstance(frame, str) else frame.get("image_path")
        if not path:
            raise RuntimeError("MineLLMPerception needs image_path. It does not call the network without a file.")
        answers = {}
        for name in self.objects:
            answers[f"present:{name}"] = parse_yes(post_image(self.url, path, TEMPLATES["present"].format(name=name)))
        for key in ("biome", "time", "weather", "brightness"):
            answers[key] = post_image(self.url, path, TEMPLATES[key])
        return scene_from_answers(answers, self.objects, self.labels)
