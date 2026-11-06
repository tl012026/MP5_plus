"""Navigation policies. C3 is a placeholder."""

from lfnav.policy.base import Policy
from lfnav.policy.c0 import RandomWalk
from lfnav.policy.c1 import RuleBased

__all__ = ["Policy", "RandomWalk", "RuleBased"]
