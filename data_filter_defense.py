import json
from vllm import LLM, SamplingParams
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.inference_utils import robust_parse
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string
from agentdojo.functions_runtime import Env, EmptyEnv, FunctionsRuntime

from .tool_execution import tool_result_to_str

SYSTEM_PROMPT = """You are a data filter designed to clean and sanitize input data..."""

def format_prompt(user_input: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_input}\n<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

def recursive_filter(obj, filter_model, instruction):
    """Apply the filter to the current object, the input object can be a dict, list, or string"""
    if isinstance(obj, dict): return {k: recursive_filter(v, filter_model, instruction) for k, v in obj.items()}
    elif isinstance(obj, list): return [recursive_filter(v, filter_model, instruction) for v in obj]
    elif isinstance(obj, str): return apply_filter_for_single(filter_model=filter_model, instruction=instruction, data=obj)
    else: return obj

def apply_filter_for_single(filter_model, instruction: str, data: str) -> str:
    """Apply data filter to given instruction and data."""
    # user_input = f"{instruction} <|end_of_instruction|> {data}"
    # prompt = format_prompt(user_input)
    
    # sampling_params = SamplingParams(
    #     temperature=0.0,
    #     max_tokens=1024,
    #     stop=["<|end_of_data|>", "<|eot_id|>"]
    # )
    return apply_filter_in_batch(filter_model, [instruction], [data])[0]


def apply_filter_in_batch(filter_model, instructions: list, datas: list) -> list:
    """Apply data filter in batch. The datas should be a list of strings. It does not work for json or list data."""
    prompts = [format_prompt(f"{instr} <|end_of_instruction|> {data}") for instr, data in zip(instructions, datas)]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["<|end_of_data|>"]
    )
    
    outputs = filter_model.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]



class DataFilterDefense(BasePipelineElement):
    """Defense element that cleans tool outputs using the DataFilter model."""
    def __init__(self, model_path: str):
        self.filter_model = LLM(model=model_path, tensor_parallel_size=1, dtype="bfloat16")

    def _apply_filter(self, instruction: str, data: str) -> str:
        user_input = f"{instruction} <|end_of_instruction|> {data}"
        prompt = format_prompt(user_input)
        params = SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|eot_id|>"])
        out = self.filter_model.generate([prompt], params)
        return out[0].outputs[0].text.strip()

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: list[ChatMessage] = [],
        extra_args: dict = {},
    ):
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args
    
        for msg in messages:
            if msg["role"] == "tool" and "content" in msg:
                try:
                    # extract user instruction
                    user_instruction = ""
                    for m in messages:
                        if m["role"] == "user":
                            user_instruction = m["content"][0]["content"]
                            break
    
                    raw_data = msg["content"][0]["content"]
    
                    json_data = robust_parse(raw_data)
                    cleaned = recursive_filter(
                        json_data,
                        filter_model=self.filter_model,
                        instruction=user_instruction
                    )
                    cleaned_str = json.dumps(cleaned, indent=2, ensure_ascii=False)

                    print("\nUser instruction:", user_instruction)
                    print("\n\n\n\n---------------------------------------------------------")
                    print("Tool call result (raw):")
                    print(raw_data)
                    print("\n\n\n\nTool call result (cleaned):")
                    print(cleaned_str)

                    msg["content"] = [text_content_block_from_string(cleaned_str)]

                except Exception as e:
                    print(f"[DataFilterDefense] Skipped cleaning due to error: {e}")
                    continue
        return query, runtime, env, messages, extra_args