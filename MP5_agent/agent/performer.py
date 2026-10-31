from utils import *
from structured_actions import *

class Performer:
    def __init__(
        self,
        memory, 
        percipient, 
        checker,
        event_logger=None,
    ):
        self.memory = memory
        self.percipient = percipient
        self.checker = checker
        self.event_logger = event_logger

    def _steps(self, env):
        return getattr(env, "steps", None)

    def _emit_action(self, env, name, args, start_step, result):
        if self.event_logger:
            self.event_logger.emit(
                "action",
                name=name,
                args=args,
                start_step=start_step,
                end_step=self._steps(env),
                result=result,
            )

    def _prep(self, env, name, args, start_step):
        check_result = self.checker.check_action_preparation(name, self.memory.inventory, args)
        if self.event_logger:
            self.event_logger.emit(
                "patroller",
                kind="action_prep",
                action=name,
                success=check_result["success"],
                reason=check_result.get("feedback", ""),
            )
        if not check_result["success"]:
            self._emit_action(env, name, args, start_step, "precheck_fail")
        return check_result

    def check_and_execute_workflow(self, env, workflow_dict, task_information, underground):
        for step in workflow_dict['workflow']:
            times = int(step['times'])
            mine_finish = False
            
            for _ in range(times):
                if mine_finish:
                    break

                for action in step['actions']:
                    name, args = action['name'], action['args']
                    start_step = self._steps(env)
                    try:
                        if name == "find":
                            check_result = self._prep(env, "find", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground
                            
                            obj = args["obj"]
                            find_obj = update_find_obj_name(obj)
                            explore_above_ground(env=env, object=find_obj, percipient=self.percipient, memory=self.memory, task_information=task_information, underground=underground)
                        
                        elif name == "move_to":
                            check_result = self._prep(env, "move_to", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground

                            obj = args["obj"]
                            approach(env=env, object=obj, underground=underground)

                        elif name == "craft":
                            check_result = self._prep(env, "craft", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground
                            
                            craft_name = list(args["obj"].keys())[0].replace(" ", "_")
                            craft_num = int(list(args["obj"].values())[0])

                            craft_num = update_craft_num(craft_name, craft_num)
                            inventory_name_list, inventory_num_list = action_craft(env, craft_name, args["platform"]=="crafting table",args["platform"]=="furnace",craft_num=craft_num)
                            self.memory.update_inventory(count_inventory(inventory_name_list, inventory_num_list))

                        elif name == "mine":
                            check_result = self._prep(env, "mine", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground

                            obj = args["obj"]
                            inventory_obj = update_inventory_obj_name(obj)

                            tool = "" if args["tool"] is None else args["tool"]

                            inventory_name_list, inventory_num_list = mine(env=env, target=obj, equipment=tool, underground=underground)
                            self.memory.update_inventory(count_inventory(inventory_name_list, inventory_num_list))
                            
                            if inventory_obj in self.memory.inventory.keys() and int(self.memory.inventory[inventory_obj]) >= times:
                                mine_finish = True

                        elif name == "fight":
                            check_result = self._prep(env, "fight", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground
                        
                        elif name == "equip":
                            check_result = self._prep(env, "equip", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground


                        elif name == "dig_down":
                            check_result = self._prep(env, "dig_down", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground

                            underground = True
                            tool = "" if args["tool"] is None else args["tool"]
                            go_down_to_y_level(env,args["y_level"],equipment = tool)


                        elif name == "dig_up":
                            check_result = self._prep(env, "dig_up", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground

                            underground = False
                        
                        elif name == "apply":
                            check_result = self._prep(env, "apply", args, start_step)
                            if not check_result["success"]:
                                return check_result, underground
                        self._emit_action(env, name, args, start_step, "ok")
                    except Exception:
                        self._emit_action(env, name, args, start_step, "exception")
                        raise
        
        check_result = {
                        "feedback": f"",
                        "success": True,
                        "suggestion": f""
                    }
        return check_result, underground
