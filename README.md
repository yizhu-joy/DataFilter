# DataFilter

[![arXiv](https://img.shields.io/badge/arXiv-2510.19207-b31b1b.svg)](https://arxiv.org/abs/2510.19207)
[![HuggingFace](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/JoyYizhu/DataFilter)

A defense system designed to protect LLM agent systems against prompt injection attacks. DataFilter provides robust protection while maintaining system utility and performance.

<p align="center">
  <img src="img/framework.png" alt="DataFilter Framework" width="60%"/>
  <img src="img/results.png" alt="DataFilter Results" width="35%"/>
  <br>
</p>

## Table of Contents

- [Quick Start](#quick-start)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Training from Scratch](#training-from-scratch)
- [Citation](#citation)

## Quick Start

### Installation

```bash
conda create -n py312vllm python=3.12
conda activate py312vllm
pip install vllm pandas 'accelerate>=0.26.0' deepspeed datasets==2.20.0
git clone https://github.com/yizhu-joy/DataFilter.git
```

### Testing the Pre-trained Model

To test our pre-trained DataFilter model:

1. **Download the model from HuggingFace:**
   ```bash
   git lfs install
   git clone https://huggingface.co/JoyYizhu/DataFilter models/DataFilter
   ```

2. **Run DataFilter inference demo:**
   ```bash
   python filter_inference.py
   ```

## Reproducing Paper Results

This section provides step-by-step instructions to reproduce all benchmark results from our paper.

### Prerequisites

Ensure you have completed the installation steps above, then authenticate with required services:

```bash
# Authenticate with HuggingFace
huggingface-cli login

# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key"

# Configure OpenAI in the config file
# Edit data/openai_config.yaml with your API key
```

### Benchmark 1: AlpacaEval2

**Metric:** Utility (Higher is Better)

Install AlpacaEval:
```bash
pip install alpaca-eval
```

**Without Defense:**
```bash
# Llama-3.1-8B
python test.py --attack none --defense none \
  --test_data data/davinci_003_outputs.json \
  -m meta-llama/Llama-3.1-8B-Instruct

# GPT-4o
python test.py --attack none --defense none \
  --test_data data/davinci_003_outputs.json \
  -m gpt-4o-2024-05-13
```

**With DataFilter Defense:**
```bash
# Llama-3.1-8B
python test.py --attack none --defense datafilter \
  --test_data data/davinci_003_outputs.json \
  -m meta-llama/Llama-3.1-8B-Instruct

# GPT-4o
python test.py --attack none --defense datafilter \
  --test_data data/davinci_003_outputs.json \
  -m gpt-4o-2024-05-13
```

### Benchmark 2: SEP

**Metric:** Attack Success Rate (ASR) - Lower is Better

**Without Defense:**
```bash
# Llama-3.1-8B
python test.py --attack completion --defense none \
  --test_data data/SEP_1000_keyword.json \
  -m meta-llama/Llama-3.1-8B-Instruct

# GPT-4o
python test.py --attack completion --defense none \
  --test_data data/SEP_1000_keyword.json \
  -m gpt-4o-2024-05-13
```

**With DataFilter Defense:**
```bash
# Llama-3.1-8B
python test.py --attack completion --defense datafilter \
  --test_data data/SEP_1000_keyword.json \
  -m meta-llama/Llama-3.1-8B-Instruct

# GPT-4o
python test.py --attack completion --defense datafilter \
  --test_data data/SEP_1000_keyword.json \
  -m gpt-4o-2024-05-13
```

### Benchmark 3: InjecAgent

**Metric:** Attack Success Rate (ASR) - Lower is Better

**Setup:**
```bash
git clone https://github.com/uiuc-kang-lab/InjecAgent.git
```

**Without Defense:**
```bash
# Llama-3.1-8B
python test_InjecAgent.py --defense none \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct

# GPT-4o
python test_InjecAgent.py --defense none \
  --model_name_or_path gpt-4o-2024-05-13
```

**With DataFilter Defense:**
```bash
# Llama-3.1-8B
python test_InjecAgent.py --defense datafilter \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct

# GPT-4o
python test_InjecAgent.py --defense datafilter \
  --model_name_or_path gpt-4o-2024-05-13
```

### Benchmark 4: AgentDojo

**Metrics:** Attack Success Rate (ASR) - Lower is Better, Utility - Higher is Better

**Setup:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ethz-spylab/agentdojo.git
   ```

2. **Configure tool output format:**
   ```bash
   sed -i \
     "s|tool_output_format: Literal\[\"yaml\", \"json\"\] \| None = None,|tool_output_format: Literal\[\"yaml\", \"json\"\] \| None = \"json\",|g" \
     agentdojo/src/agentdojo/scripts/benchmark.py
   ```

3. **Update tool result formatting:**
   
   Replace the `tool_result_to_str` function in `agentdojo/src/agentdojo/agent_pipeline/tool_execution.py` with:

   ```python
   import datetime
   from typing import Any
   import json

   def tool_result_to_str(
       tool_result: Any, dump_fn: Callable[[dict | list[dict]], str] | None = None,
   ) -> str:
       """Format tool results into a string. 
       """
       if dump_fn is None:
           dump_fn = lambda x: yaml.safe_dump(x, sort_keys=False)
       elif dump_fn is json.dumps:
           dump_fn = lambda x: json.dumps(x, indent=2, default=str)

       if isinstance(tool_result, BaseModel):
           return dump_fn(tool_result.model_dump()).strip()

       if isinstance(tool_result, list):
           res_items = []
           for item in tool_result:
               if isinstance(item, (str, int)):
                   res_items.append(str(item))
               elif isinstance(item, BaseModel):
                   res_items.append(item.model_dump())
               elif isinstance(item, datetime.datetime):
                   res_items.append(item.isoformat())
               else:
                   raise TypeError(f"Not valid type for item tool result: {type(item)}")
           return dump_fn(res_items).strip()
       
       if isinstance(tool_result, datetime.datetime):
           return tool_result.isoformat()

       return str(tool_result)
   ```

4. **Install DataFilter defense:**
   ```bash
   # Copy defense files
   cp data_filter_defense.py agentdojo/src/agentdojo/agent_pipeline/
   cp inference_utils.py agentdojo/src/agentdojo/agent_pipeline/
   ```

5. **Register DataFilter defense:**
   
   Edit `agentdojo/src/agentdojo/agent_pipeline/agent_pipeline.py`:

   a. Add import at the top:
   ```python
   from agentdojo.agent_pipeline.data_filter_defense import DataFilterDefense
   ```

   b. Update the `DEFENSES` list (around line 43):
   ```python
   DEFENSES = [
       "tool_filter",
       "transformers_pi_detector",
       "spotlighting_with_delimiting",
       "repeat_user_prompt",
       "data_filter",
   ]
   ```

   c. Add at the bottom of the file:
   ```python
   if config.defense == "data_filter":
       data_filter_element = DataFilterDefense(
           model_path="/path/to/DataFilter/checkpoint-300"
       )
       tools_loop = ToolsExecutionLoop(
           [ToolsExecutor(tool_output_formatter), data_filter_element, llm]
       )
       pipeline = cls([system_message_component, init_query_component, llm, tools_loop])
       pipeline.name = f"{llm_name}-{config.defense}"
       return pipeline
   ```

**Run Benchmark:**
```bash
cd agentdojo/src
python -m agentdojo.scripts.benchmark \
  --model GPT_4O_2024_05_13 \
  --defense data_filter \
  --attack tool_knowledge
```

## Training from Scratch

To train your own DataFilter model:

### Step 1: Download Base Dataset

Download the Alpaca cleaned dataset from [HuggingFace](https://huggingface.co/datasets/yahma/alpaca-cleaned) and place it in the `data/` directory.

### Step 2: Generate Training Data

Generate synthetic attack and benign examples:

```bash
python generate_data.py --train --position --cut_benign \
  --attack_types Completion Naive Ignore
```

### Step 3: Train the Model

Train using distributed training with DeepSpeed:

```bash
torchrun --nproc_per_node=2 train.py --deepspeed ds_config.json
```

Adjust `--nproc_per_node` based on your available GPUs.

## Citation

If you use DataFilter in your research, please cite our paper:

```bibtex
@misc{wang2025datafilter,
  title={Defending Against Prompt Injection with DataFilter}, 
  author={Yizhu Wang and Sizhe Chen and Raghad Alkhudair and Basel Alomair and David Wagner},
  year={2025},
  eprint={2510.19207},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2510.19207}, 
}
```

## License

Please refer to the LICENSE file for details.
