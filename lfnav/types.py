"""JSON-serializable dataclasses shared by perception, matching, policy and judging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k != "_path"}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


@dataclass
class TaskSpec:
    id: str
    source: str
    description: str
    target: str
    conditions: dict = field(default_factory=dict)
    episode: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        return cls(
            id=str(data["id"]),
            source=str(data.get("source", "")),
            description=str(data.get("description", "")),
            target=str(data["target"]),
            conditions=dict(data.get("conditions") or {}),
            episode=dict(data.get("episode") or {}),
        )


@dataclass
class ObjectBelief:
    present: float = 0.0
    direction: dict = field(default_factory=lambda: {"left": 0.0, "center": 0.0, "right": 0.0})
    distance: dict = field(default_factory=lambda: {"near": 0.0, "mid": 0.0, "far": 0.0})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectBelief":
        return cls(
            present=float(data.get("present", 0.0)),
            direction=dict(data.get("direction") or {}),
            distance=dict(data.get("distance") or {}),
        )


@dataclass
class SceneState:
    objects: dict = field(default_factory=dict)
    biome: dict = field(default_factory=dict)
    time: dict = field(default_factory=dict)
    weather: dict = field(default_factory=dict)
    brightness: dict = field(default_factory=dict)
    unknown: list = field(default_factory=list)
    ignore: list = field(default_factory=list)
    patches: dict = field(default_factory=dict)

    def object(self, name: str) -> ObjectBelief:
        raw = self.objects.get(name)
        if raw is None:
            return ObjectBelief()
        if isinstance(raw, ObjectBelief):
            return raw
        return ObjectBelief.from_dict(raw)

    def to_dict(self) -> dict:
        objects = {}
        for name, belief in self.objects.items():
            objects[name] = belief.to_dict() if isinstance(belief, ObjectBelief) else dict(belief)
        return {
            "objects": objects,
            "biome": dict(self.biome),
            "time": dict(self.time),
            "weather": dict(self.weather),
            "brightness": dict(self.brightness),
            "unknown": list(self.unknown),
            "ignore": [list(row) for row in (self.ignore or [])],
            "patches": {name: [list(row) for row in grid] for name, grid in (self.patches or {}).items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneState":
        objects = {k: ObjectBelief.from_dict(v) for k, v in (data.get("objects") or {}).items()}
        return cls(
            objects=objects,
            biome=dict(data.get("biome") or {}),
            time=dict(data.get("time") or {}),
            weather=dict(data.get("weather") or {}),
            brightness=dict(data.get("brightness") or {}),
            unknown=list(data.get("unknown") or []),
            ignore=[list(row) for row in (data.get("ignore") or [])],
            patches={name: [list(row) for row in grid] for name, grid in (data.get("patches") or {}).items()},
        )


@dataclass
class MatchResult:
    status: str
    stop: bool
    satisfied: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    target_direction: str | None = None
    target_distance: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        return cls(
            status=str(data["status"]),
            stop=bool(data["stop"]),
            satisfied=dict(data.get("satisfied") or {}),
            missing=list(data.get("missing") or []),
            target_direction=data.get("target_direction"),
            target_distance=data.get("target_distance"),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class PolicyInput:
    frame_features: list | None = None
    task: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)
    scene: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    recent_actions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyInput":
        return cls(
            frame_features=data.get("frame_features"),
            task=dict(data.get("task") or {}),
            match=dict(data.get("match") or {}),
            scene=dict(data.get("scene") or {}),
            memory=dict(data.get("memory") or {}),
            recent_actions=list(data.get("recent_actions") or []),
        )


@dataclass
class PolicyOutput:
    action_chunk: list = field(default_factory=list)
    waypoints: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.action_chunk and not self.waypoints:
            raise ValueError("PolicyOutput needs action_chunk or waypoints")

    def to_dict(self) -> dict:
        return {"action_chunk": list(self.action_chunk), "waypoints": list(self.waypoints)}

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyOutput":
        return cls(
            action_chunk=list(data.get("action_chunk") or []),
            waypoints=list(data.get("waypoints") or []),
        )
