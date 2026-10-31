from utils import *
from planner import Planner
from patroller import Patroller
from memory import Memory
from performer import Performer
from percipient import Percipient
from config import load_config, DEFAULT_CONFIG_PATH, resolve_api_key
from llm_factory import get_openai_key
from runtime import runtime
from event_logger import EventLogger
from env_wrapper import EnvWrapper, EpisodeTimeout, EpisodeDeath
from structured_actions import reset_state
from success import classify_task_kind, inventory_has_item, snapshot_obs, failure_category
import argparse
import os
import numpy as np
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent


def make_minedojo_env(config, seed):
    import minedojo
    e = config["env"]
    vradius = e["voxel_radius"]
    kwargs = dict(
        task_id=e["task_id"],
        target_names=e["target_names"],
        image_size=tuple(e["image_size"]),
        target_quantities=e["target_quantities"],
        seed=e["env_seed"],
        spawn_rate=e["spawn_rate"],
        break_speed_multiplier=e["break_speed_multiplier"],
        spawn_range_low=tuple(e["spawn_range_low"]),
        spawn_range_high=tuple(e["spawn_range_high"]),
        start_at_night=e["start_at_night"],
        world_seed=seed,
        use_voxel=e["use_voxel"],
        voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius,
                        xmax=vradius, ymax=vradius, zmax=vradius),
        use_lidar=e["use_lidar"],
    )
    if e.get("specified_biome") is not None:
        kwargs["specified_biome"] = e["specified_biome"]
    if e["use_lidar"]:
        p0, p1, ps = e["lidar_pitch"]
        y0, y1, ys = e["lidar_yaw"]
        kwargs["lidar_rays"] = [
            (np.pi * pitch / 180, np.pi * yaw / 180, e["lidar_range"])
            for pitch in np.arange(p0, p1, ps)
            for yaw in np.arange(y0, y1, ys)
        ]
    return minedojo.make(**kwargs)


def apply_start_inventory(env, config):
    from minedojo.sim import InventoryItem
    items = config["env"].get("start_inventory") or []
    if not items:
        return
    env.set_inventory([
        InventoryItem(slot=it["slot"], name=it["name"], variant=it.get("variant"), quantity=it["quantity"])
        for it in items
    ])


def run_episode(task_file, seed, config, out_dir, env=None):
    """
    Run one task file as one episode. The planner-performer-patroller loop is unchanged.
    env: optional already-created env (tests inject a fake). Otherwise MineDojo is constructed.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # load_prompt() and task relative paths assume cwd is MP5_agent/agent/
    prev_cwd = os.getcwd()
    os.chdir(AGENT_DIR)
    # OUT/<task>/<seed>/ artifacts; events live at OUT/<task>/<seed>.events.jsonl
    logger = EventLogger(out_dir.parent / f"{out_dir.name}.events.jsonl")
    runtime.bind(out_dir, config["eval"], event_logger=logger)
    reset_state()

    log_path = runtime.logs_dir / "agent.log"
    logging.basicConfig(filename=str(log_path), filemode="w", level=logging.INFO, format="%(message)s", force=True)

    started = time.time()
    kind = classify_task_kind(task_file)
    result = {
        "task_file": str(task_file),
        "task_kind": kind,
        "seed": seed,
        "success": None,
        "outcome": None,
        "steps": 0,
        "elapsed_s": None,
        "self_reported_steps": None,
        "obs_snapshot": None,
        "failure_category": None,
        "llm_counts": None,
        "inventory": {},
    }
    last_action_find = False
    planner_parse_failed = False
    retries_exhausted = False
    own_env = env is None
    wrapped = None

    try:
        openai_key = get_openai_key()
        if env is None:
            raw_env = make_minedojo_env(config, seed)
            raw_env.reset()
            apply_start_inventory(raw_env, config)
            events, _, _, _ = raw_env.step(list(config["env"]["warmup_action"]))
        else:
            raw_env = env
            raw_env.reset()
            events, _, _, _ = raw_env.step(list(config["env"]["warmup_action"]))

        wrapped = EnvWrapper(raw_env, config["eval"]["max_steps"], logger=logger)
        env = wrapped
        log_info(seed)

        underground = False
        memory = Memory(
            openai_key=openai_key,
            use_history_workflow=False,
            ckpt_dir=str(runtime.out_dir),
            model_cfg=config["models"]["memory"],
        )
        patroller = Patroller(
            openai_key=openai_key, memory=memory,
            model_cfg=config["models"]["patroller"], event_logger=logger,
        )
        planner = Planner(
            openai_key=openai_key, memory=memory,
            model_cfg=config["models"]["planner"], event_logger=logger,
        )
        perc = config["percipient"]
        percipient = Percipient(
            openai_key=openai_key, memory=memory,
            question_model_cfg=config["models"]["percipient_question"],
            answer_method=perc["answer_method"],
            answer_model=perc["answer_model"],
            answer_mllm_url=perc.get("mllm_url") or None,
            answer_gpt_name=perc.get("gpt_vision_model"),
            gpt_vision_base_url=perc.get("gpt_vision_base_url"),
            event_logger=logger,
        )
        share_memory(memory=memory, events=events)

        with open(task_file, "r") as f:
            task_list = json.load(f)

        all_subgoals_done = True
        for task_id, task_information in enumerate(task_list[::-1]):
            every_task_max_retries = int(config["eval"]["max_subgoal_retries"])
            check_result = {}
            logger.emit("subgoal", phase="start", task=task_information.get("task"), index=task_id)
            log_info(f"Task: { task_information['description'] }")

            while every_task_max_retries >= 0:
                memory.reset_current_environment_information()
                log_info(f"My inventory: {memory.inventory}")

                if every_task_max_retries == 0:
                    log_info("************Failed to complete this task. Consider updating your prompt.************\n\n")
                    retries_exhausted = True
                    all_subgoals_done = False
                    break

                workflow_dict = planner.get_workflow(task_information=task_information, underground=underground, check_result=check_result)
                if not workflow_dict:
                    planner_parse_failed = True

                performer = Performer(memory=memory, percipient=percipient, checker=patroller, event_logger=logger)
                last_action_find = _workflow_has_find(workflow_dict)
                check_result, underground = performer.check_and_execute_workflow(
                    env=env, workflow_dict=workflow_dict, task_information=task_information, underground=underground
                )
                if not check_result["success"]:
                    log_info(f"Action Preparation Failure: {check_result}")
                    every_task_max_retries -= 1
                    continue

                check_result = patroller.check_task_success(task_information=task_information)

                if check_result["success"]:
                    memory.add_successful_workflow(task_information["task"], task_information["description"], workflow_dict["workflow"])
                    logger.emit("subgoal", phase="end", task=task_information.get("task"), retries_used=int(config["eval"]["max_subgoal_retries"]) - every_task_max_retries, success=True)
                    break
                else:
                    every_task_max_retries -= 1
                    continue
            else:
                all_subgoals_done = False

            if retries_exhausted:
                break

        result["inventory"] = dict(memory.inventory)
        memory.reset_all()
        if all_subgoals_done and not retries_exhausted:
            log_info("############ Successfully Finish All Tasks ############")
            if kind == "process":
                final = task_list[0]
                ok = inventory_has_item(result["inventory"], final)
                result["success"] = bool(ok)
                result["outcome"] = "success" if ok else "fail"
            else:
                result["success"] = None
                result["outcome"] = "self_reported"
                result["self_reported_steps"] = env.steps
        else:
            result["success"] = False if kind == "process" else None
            result["outcome"] = "fail"

        result["obs_snapshot"] = snapshot_obs(getattr(env, "last_obs", None))

    except EpisodeTimeout as exc:
        result["outcome"] = "timeout"
        result["success"] = False if kind == "process" else None
        result["steps"] = exc.steps
    except EpisodeDeath as exc:
        result["outcome"] = "death"
        result["success"] = False if kind == "process" else None
        result["steps"] = exc.steps
    except Exception as exc:
        result["outcome"] = "error"
        result["success"] = False if kind == "process" else None
        result["error"] = repr(exc)
        log_info(f"Episode error: {exc}")
    finally:
        result["steps"] = result["steps"] or getattr(env, "steps", 0) if env is not None else 0
        result["elapsed_s"] = time.time() - started
        result["llm_counts"] = dict(logger.counts)
        result["failure_category"] = failure_category(
            result["outcome"], last_action_find, planner_parse_failed, retries_exhausted
        )
        logger.emit(
            "episode_end",
            outcome=result["outcome"],
            success=result["success"],
            steps=result["steps"],
            elapsed_s=result["elapsed_s"],
            llm_counts=result["llm_counts"],
            failure_category=result["failure_category"],
        )
        logger.close()
        if own_env and env is not None:
            closer = getattr(getattr(env, "_env", env), "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        runtime.reset()
        reset_state()
        os.chdir(prev_cwd)
    return result


def _workflow_has_find(workflow_dict):
    if not isinstance(workflow_dict, dict):
        return False
    for step in workflow_dict.get("workflow") or []:
        for action in step.get("actions") or []:
            if action.get("name") == "find":
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--mllm_url", type=str, default="")
    parser.add_argument("--openai_key", type=str, default="", help="Ignored; use OPENAI_API_KEY")
    parser.add_argument("--gpt_model_name", type=str, default="")
    parser.add_argument("--answer_method", type=str, default="")
    parser.add_argument("--answer_model", type=str, default="")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mllm_url:
        config["percipient"]["mllm_url"] = args.mllm_url
    if args.answer_method:
        config["percipient"]["answer_method"] = args.answer_method
    if args.answer_model:
        config["percipient"]["answer_model"] = args.answer_model
    if args.gpt_model_name:
        for role in ("planner", "patroller", "percipient_question"):
            config["models"][role]["model_name"] = args.gpt_model_name

    seed = args.seed if args.seed is not None else random.randint(1, 1000000000000)
    out_dir = args.out_dir or str(Path("runs") / Path(args.task).stem / str(seed))
    resolve_api_key()
    run_episode(args.task, seed, config, out_dir)


if __name__ == "__main__":
    main()
