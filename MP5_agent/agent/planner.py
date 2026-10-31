import time
from utils import *
from llm_factory import create_chat_openai

class Planner:
    def __init__(
        self,
        openai_key,
        memory,
        model_name="gpt-4-0613",
        temperature=0,
        model_cfg=None,
        event_logger=None,
    ):
        if model_cfg is None:
            model_cfg = {"model_name": model_name, "base_url": None, "temperature": temperature}
        self.llm = create_chat_openai(model_cfg)
        self.memory = memory
        self.event_logger = event_logger
        assert  self.memory is not None, "Please input memory"
    
    def get_workflow(self, task_information, underground, check_result, max_retries=5):

        if max_retries == 0:
            log_info("************Failed to get workflow. Consider updating your prompt.************\n\n")
            return {}

        try:
            structured_action_system = load_prompt("structured_action_system")
            task_information_string = dict_to_prompt(task_information)
            current_environment_information=""
            
            if underground:
                current_environment_information += "- position: underground\n"
            else:
                current_environment_information += "- position: ground\n"

            reference_plan_string = list_dict_to_prompt(self.memory.seach_workflows(task_information["description"]))

            structured_action_query = load_prompt("structured_action_query").format(
                task_information=task_information_string, 
                current_environment_information=current_environment_information,
                inventory=self.memory.inventory,
                reference_plan=reference_plan_string
            )


            if len(check_result) == 0:
                structured_action_query += "\nPlan your workflow. Remember to follow the response format."
            else:
                structured_action_query += f"""The previous workflow failed. 
                The reason for the failure: {check_result["feedback"]}.
                A suggested recommendations: {check_result["suggestion"]}. 
                re-plan your workflow. Remember to follow the response format."""
                
            messages = [
                SystemMessage(content=structured_action_system),
                HumanMessage(content=structured_action_query)
            ]

            t0 = time.time()
            raw = self.llm(messages)
            latency = time.time() - t0
            workflow_dict = raw.content
            log_info(f"Create Workflow Result: {workflow_dict}")
            parse_ok = True
            try:
                parsed = fix_and_parse_json(workflow_dict)
            except Exception:
                parse_ok = False
                parsed = None
            if self.event_logger:
                self.event_logger.emit(
                    "planner_call",
                    subgoal=task_information.get("task") if isinstance(task_information, dict) else None,
                    inventory_summary=dict(self.memory.inventory) if self.memory.inventory is not None else {},
                    feedback_summary=(check_result or {}).get("feedback", ""),
                    raw_output=workflow_dict,
                    parse_ok=parse_ok,
                    retries_left=max_retries,
                    latency_s=latency,
                    tokens=_token_count(raw),
                )
            if not parse_ok:
                raise ValueError("planner JSON parse failed")
            return parsed
        except Exception as e:
            log_info(f"Error arises in Plan Workflow part: {e} Trying again!\n\n")
            self.memory.reset_current_environment_information()

            return self.get_workflow(
                task_information, 
                underground, 
                check_result, 
                max_retries=max_retries - 1
            )


def _token_count(raw):
    for attr in ("response_metadata", "llm_output"):
        meta = getattr(raw, attr, None)
        if isinstance(meta, dict):
            usage = meta.get("token_usage") or meta.get("usage")
            if isinstance(usage, dict):
                return usage.get("total_tokens")
    return None