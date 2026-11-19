"""General process-task interface with explicit, incrementally extended knowledge.

Model output is in execution order. Saved MP5 tasks are reversed exactly once.
This is a new implementation, not an exact reproduction of the paper's Parser.
"""
import json
import math

from pathlib import Path

DEFAULT_KNOWLEDGE = Path(__file__).resolve().parent / "knowledge/process.json"


class KnowledgeGap(ValueError):
    """A goal or dependency has no reviewed knowledge / executor support."""


def load_knowledge(path=None):
    data = json.loads(Path(path or DEFAULT_KNOWLEDGE).read_text())
    return data["records"]


def select_knowledge(goals, records):
    """Exact item lookup plus dependency closure; not embedding-based RAG."""
    by_output = {r["output_item"]: name for name, r in records.items()}
    selected = {}
    visiting = set()

    def visit(item):
        name = by_output.get(item)
        if name is None:
            raise KnowledgeGap(f"needs_knowledge: {item}")
        if name in selected:
            return
        if name in visiting:
            raise ValueError(f"Cyclic knowledge dependency: {name}")
        record = records[name]
        if not record.get("enabled", False):
            raise KnowledgeGap(f"needs_executor_support: {name}")
        visiting.add(name)
        for dependency in list((record["material"] or {}).keys()) + [
                v for v in (record["tool"], record["platform"]) if v]:
            visit(dependency)
        visiting.remove(name)
        selected[name] = record

    for goal in goals:
        visit(goal)
    return selected


SYSTEM_PROMPT = """Decompose a Minecraft objective into item subgoals, not actions.
Return ONLY a JSON array in EXECUTION ORDER, prerequisites first, final goal last.
Each object must contain: task, quantity, material, tool, platform, tips, description.
Use exactly the task names from supplied knowledge. quantity is a positive integer.
quantity means items produced by THIS subgoal, not the cumulative inventory total.
material is the total ingredients consumed, accounting for whole crafting batches.
tool and platform must match knowledge; they must be available and are not consumed.
tips and description are nonempty English strings matching the quantities.
Before writing the array, work backwards from ALL final goals:
1. Expand material, tool AND platform dependencies recursively. A platform named
   in a recipe is a required inventory item, NOT an automatically available service.
2. If a required tool/platform is absent initially, add its acquisition subgoal
   before use and include the materials consumed to create it.
3. Sum shared ingredient demand across ALL subgoals, including infrastructure,
   then round production up to whole recipe batches. Never plan only one batch
   of each ingredient without checking total downstream demand.
4. Simulate the inventory forward: subtract consumed ingredients at each step,
   retain surplus, and verify all final goals still hold.
Plan enough resources for ALL downstream consumption. Surplus output is retained.
Initial inventory and final inventory goals are supplied separately. All goals must
remain satisfied after execution. Put a requested goal last for MP5 compatibility.
Only use the supplied knowledge; do not invent recipes or execution capabilities.
An empty array is allowed when initial inventory already satisfies all goals.
"""


def check_counts(counts, label, allow_zero=False):
    if not isinstance(counts, dict) or any(
        not isinstance(k, str) or not k or type(v) is not int or v < (0 if allow_zero else 1)
        for k, v in counts.items()
    ):
        raise ValueError(f"{label} must map item names to integer counts")


def validate_execution_tasks(tasks, *, goals=None, initial_inventory=None, knowledge=None):
    """Validate known recipes, dependencies and final inventory; no game claims."""
    goals = {"wooden pickaxe": 1} if goals is None else goals
    check_counts(goals, "goals")
    if not goals:
        raise ValueError("At least one final goal is required")
    records = load_knowledge() if knowledge is None else knowledge
    select_knowledge(goals, records)
    inventory = dict(initial_inventory or {})
    check_counts(inventory, "initial inventory", allow_zero=True)
    required = {"task", "quantity", "material", "tool", "platform", "tips", "description"}
    if not isinstance(tasks, list):
        raise ValueError("Expected an array of item subgoals")
    trace = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or not required <= task.keys():
            raise ValueError(f"Subgoal {index}: missing task fields")
        name, qty = task["task"], task["quantity"]
        if not isinstance(name, str) or name not in records:
            raise KnowledgeGap(f"Subgoal {index}: needs_knowledge: {name!r}")
        record = records[name]
        if not record.get("enabled", False):
            raise KnowledgeGap(f"needs_executor_support: {name}")
        if type(qty) is not int or qty < 1:
            raise ValueError(f"Subgoal {index}: quantity must be a positive integer")
        if task["tool"] != record["tool"]:
            raise ValueError(f"{name}: tool must be {record['tool']!r}")
        for dependency in (record["tool"], record["platform"]):
            if dependency and inventory.get(dependency, 0) < 1:
                raise ValueError(f"{name}: missing preceding {dependency} subgoal")
        if any(not isinstance(task[k], str) or not task[k].strip() for k in ("tips", "description")):
            raise ValueError(f"Subgoal {index}: nonempty tips and description required")
        if record["kind"] == "mine":
            if task["material"] is not None or task["platform"] != record["platform"]:
                raise ValueError(f"{name}: incorrect material/platform")
            output_item = record["output_item"]
            inventory[output_item] = inventory.get(output_item, 0) + qty
        elif record["kind"] == "craft":
            recipe, output, platform = record["material"], record["output_count"], record["platform"]
            batches = math.ceil(qty / output)
            materials = {item: count * batches for item, count in recipe.items()}
            if (not isinstance(task["material"], dict)
                    or any(type(v) is not int for v in task["material"].values())
                    or task["material"] != materials):
                raise ValueError(f"{name}: material must be {materials} for quantity {qty}")
            if task["platform"] != platform:
                raise ValueError(f"{name}: platform must be {platform!r}")
            if platform and inventory.get(platform, 0) < 1:
                raise ValueError(f"{name}: missing preceding {platform} subgoal")
            for item, count in materials.items():
                if inventory.get(item, 0) < count:
                    raise ValueError(f"{name}: insufficient {item}; need {count}, have {inventory.get(item, 0)}")
            for item, count in materials.items():
                inventory[item] -= count
            output_item = record["output_item"]
            inventory[output_item] = inventory.get(output_item, 0) + batches * output
        else:
            raise KnowledgeGap(f"Unsupported knowledge kind: {record['kind']}")
        trace.append({"task": name, "inventory_after": dict(inventory)})
    for item, count in goals.items():
        if inventory.get(item, 0) < count:
            raise ValueError(f"Final goal unsatisfied: {item} >= {count}")
    if tasks and records[tasks[-1]["task"]]["output_item"] not in goals:
        raise ValueError("Final subgoal must correspond to a requested goal")
    return trace


def parse_task(request, model_cfg, out_dir, *, goals=None, initial_inventory=None, knowledge_path=None):
    out_dir.mkdir(parents=True, exist_ok=False)
    goals = {"wooden pickaxe": 1} if goals is None else goals
    initial_inventory = dict(initial_inventory or {})
    (out_dir / "request.json").write_text(json.dumps({
        "request": request, "initial_inventory": initial_inventory, "goals": goals,
        "model_config": model_cfg, "system_prompt": SYSTEM_PROMPT,
    }, indent=2))
    try:
        check_counts(goals, "goals")
        if not goals:
            raise ValueError("At least one final goal is required")
        check_counts(initial_inventory, "initial inventory", allow_zero=True)
        knowledge = select_knowledge(goals, load_knowledge(knowledge_path))
        (out_dir / "knowledge_used.json").write_text(json.dumps(knowledge, indent=2))
        from langchain.schema import SystemMessage, HumanMessage
        from llm_factory import create_chat_openai
        client = create_chat_openai(model_cfg)
        client.request_timeout = 60
        client.max_retries = 0
        client.max_tokens = 4000
        payload = {"request": request, "goals": goals,
                   "initial_inventory": initial_inventory, "knowledge": knowledge}
        messages = [SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload))]
        attempts = []
        for attempt in range(1, 3):  # One generation plus at most one validation repair.
            reply = client(messages)
            (out_dir / f"raw_response_{attempt}.txt").write_text(reply.content)
            (out_dir / "raw_response.txt").write_text(reply.content)
            metadata = {
                "response_metadata": getattr(reply, "response_metadata", {}),
                "additional_kwargs": getattr(reply, "additional_kwargs", {}),
            }
            (out_dir / f"response_metadata_{attempt}.json").write_text(json.dumps(metadata, indent=2, default=str))
            (out_dir / "response_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
            try:
                tasks = json.loads(reply.content)
                trace = validate_execution_tasks(tasks, goals=goals,
                    initial_inventory=initial_inventory, knowledge=knowledge)
            except ValueError as exc:
                attempts.append({"attempt": attempt, "passed": False, "error": str(exc)})
                (out_dir / "attempts.json").write_text(json.dumps(attempts, indent=2))
                if attempt == 2:
                    raise
                feedback = {
                    "previous_response": reply.content,
                    "validation_error": str(exc),
                    "instruction": "Return a complete corrected JSON array. Recheck ALL tool/platform "
                        "dependencies and total material demand, not only the first reported error. "
                        "Keep the original goals, inventory and supplied knowledge unchanged.",
                }
                (out_dir / "repair_feedback.json").write_text(json.dumps(feedback, indent=2))
                messages = messages + [HumanMessage(content=json.dumps(feedback))]
            else:
                attempts.append({"attempt": attempt, "passed": True})
                (out_dir / "attempts.json").write_text(json.dumps(attempts, indent=2))
                break

    except Exception as exc:
        (out_dir / "validation.json").write_text(json.dumps({"passed": False, "status": "needs_knowledge" if isinstance(exc, KnowledgeGap) else "invalid", "error": str(exc)}, indent=2))
        raise
    (out_dir / "execution_order.json").write_text(json.dumps(tasks, indent=2))
    (out_dir / "validation.json").write_text(json.dumps({"passed": True, "recipe_trace": trace}, indent=2))
    # Multi-goal / nonempty-start tasks need evaluation and reset adapters before execution.
    if (len(goals) != 1 or initial_inventory or not tasks
            or tasks[-1]["quantity"] != next(iter(goals.values()))):
        (out_dir / "validation.json").write_text(json.dumps({
            "passed": True, "status": "validated_plan_only", "recipe_trace": trace,
            "reason": "Runner needs one final subgoal matching the goal quantity and an empty start."
        }, indent=2))
        return out_dir / "execution_order.json"
    dest = out_dir / "process" / "generated_task.json"
    dest.parent.mkdir()
    dest.write_text(json.dumps(tasks[::-1], indent=2))
    return dest
