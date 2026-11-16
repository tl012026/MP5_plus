"""MineDreamer as the MP5 Performer.

MP5's planner emits workflow actions (find / move_to / mine / craft / dig_down ...).
This package turns each action into a MineDreamer text instruction, runs the
Imaginator -> Prompt Generator -> STEVE-1 loop until the action's done-condition
holds or its step budget runs out, and writes one formatted result file per action.

Everything here imports without MineDojo, MineRL, torch, or a GPU. The real
Imaginator is reached over HTTP; STEVE-1 and the Prompt Generator are pluggable.
"""

from minedreamer_bridge.schema import SCHEMA_ID  # noqa: F401
