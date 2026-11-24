#!/usr/bin/env python3
"""Per-subgoal breakdown of one MP5 episode's .events.jsonl (written by run_agent.py).

    python3 tools/subgoal_report.py OUT/<task>/<seed>.events.jsonl            # table
    python3 tools/subgoal_report.py OUT/<task>/<seed>.events.jsonl --detail   # + event trace

Table columns, one row per subgoal:
  steps       env steps from the end of the previous subgoal to this subgoal's last action
  step range  first action start_step to last action end_step
  planner     planner calls (1 + replans)
  retries     retries_used from the subgoal end event
  actions     executed actions (find / move_to / mine / craft ...)
  prep fail   Patroller action-preparation checks that failed
  check fail  Patroller task-success checks that failed
  percept     MineLLM / vision calls
  secs        wall time

--detail adds, per subgoal, each plan (with the feedback it received), action, failed
Patroller check (reason + suggestion) and perception question/answer.
Reads only the events file; standard library only.
"""
import json
import sys

args = [a for a in sys.argv[1:] if a != "--detail"]
if len(args) != 1:
    sys.exit("usage: python3 tools/subgoal_report.py EVENTS_JSONL [--detail]")
path, detail = args[0], "--detail" in sys.argv
events = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

segments, cur = [], None
for r in events:
    if r["event"] == "subgoal" and r.get("phase") == "start":
        cur = {"task": r.get("task"), "index": r.get("index"), "events": [], "end": None}
        segments.append(cur)
    elif r["event"] == "subgoal" and r.get("phase") == "end":
        cur["end"] = r
        cur = None
    elif cur is not None:
        cur["events"].append(r)
    elif r["event"] != "episode_end":
        print("event outside any subgoal:", r["event"])

end = next((r for r in events if r["event"] == "episode_end"), {})
prev_end_step = 0
print(f"{'subgoal':<16}{'steps':>7}{'step range':>16}{'planner':>9}{'retries':>9}{'actions':>9}{'prep fail':>11}{'check fail':>12}{'percept':>9}{'secs':>7}")
for s in segments:
    ev = s["events"]
    acts = [r for r in ev if r["event"] == "action"]
    starts = [r["start_step"] for r in acts if r.get("start_step") is not None]
    ends = [r["end_step"] for r in acts if r.get("end_step") is not None]
    lo, hi = (min(starts), max(ends)) if acts else (prev_end_step, prev_end_step)
    steps = hi - prev_end_step  # include steps between subgoals
    prev_end_step = hi
    planner = sum(r["event"] == "planner_call" for r in ev)
    prep_fail = [r for r in ev if r["event"] == "patroller" and r.get("kind") == "action_prep" and not r.get("success")]
    check = [r for r in ev if r["event"] == "patroller" and r.get("kind") == "task_success"]
    check_fail = [r for r in check if not r.get("success")]
    perc = [r for r in ev if r["event"] == "perception_call"]
    t0 = ev[0]["ts"] if ev else 0
    t1 = s["end"]["ts"] if s["end"] else (ev[-1]["ts"] if ev else 0)
    retries = s["end"].get("retries_used") if s["end"] else "n/a"
    print(f"{s['task']:<16}{steps:>7}{f'{lo}-{hi}':>16}{planner:>9}{str(retries):>9}{len(acts):>9}{len(prep_fail):>11}{len(check_fail):>12}{len(perc):>9}{t1 - t0:>7.0f}")
    if detail:
        for r in ev:
            e = r["event"]
            if e == "planner_call":
                print(f"    PLAN retries_left={r.get('retries_left')} parse_ok={r.get('parse_ok')} feedback={str(r.get('feedback_summary'))[:160]!r}")
            elif e == "action":
                print(f"    ACT  {r.get('name')} {json.dumps(r.get('args'))[:110]} steps {r.get('start_step')}->{r.get('end_step')} result={str(r.get('result'))[:90]!r}")
            elif e == "patroller" and r.get("kind") == "action_prep" and not r.get("success"):
                print(f"    PREP-FAIL {r.get('action')}: {str(r.get('reason'))[:200]!r}")
            elif e == "patroller" and r.get("kind") == "task_success":
                tag = "CHECK-OK  " if r.get("success") else "CHECK-FAIL"
                print(f"    {tag} reason={str(r.get('reason'))[:200]!r}")
                if not r.get("success"):
                    print(f"               suggestion={str(r.get('suggestion'))[:220]!r}")
            elif e == "perception_call":
                print(f"    PERC Q={r.get('question')!r}\n         A={r.get('answer')!r} ({r.get('latency_s', 0):.1f}s, {r.get('image_path', '').split('/')[-1]})")
print(f"episode_end: outcome={end.get('outcome')} success={end.get('success')} steps={end.get('steps')} elapsed={end.get('elapsed_s', 0):.0f}s")
