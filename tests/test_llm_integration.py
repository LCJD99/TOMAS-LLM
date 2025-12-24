#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Test script for LLM integration (Qwen2.5 backbone).

Tests:
1. QwenBackbone loading (0.5B for testing)
2. ContextProjector projection
3. Context injection into LLM
4. Text generation with context
5. Complete TOMAS-LLM model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml

print("=" * 70)
print("Testing LLM Integration - Qwen2.5 Backbone")
print("=" * 70)

# Load config
with open('configs/simple-test.yaml') as f:
    config = yaml.safe_load(f)

# Test 1: Load Qwen Backbone
print("\n" + "=" * 70)
print("Test 1: Load Qwen2.5-0.5B Backbone")
print("=" * 70)

from llm.qwen_backbone import QwenBackbone

print(f"Loading model: {config['llm']['model']['name']}")
qwen = QwenBackbone.from_config(config)

print(f"✓ Model loaded successfully")
print(f"  - Model: {qwen.model_name}")
print(f"  - Hidden dim: {qwen.hidden_dim}")
print(f"  - Vocab size: {qwen.vocab_size}")
print(f"  - Device: {qwen.device}")
print(f"  - Dtype: {qwen.dtype_str}")

# Test 2: ContextProjector
print("\n" + "=" * 70)
print("Test 2: Context Projector")
print("=" * 70)

from llm.qwen_backbone import ContextProjector

projector = ContextProjector.from_config(config)
print(f"✓ Projector created")
print(f"  - LLM hidden dim: {projector.llm_hidden_dim}")
print(f"  - Temporal dim: {projector.temporal_dim}")
print(f"  - Task dim: {projector.task_dim}")
print(f"  - Tool dim: {projector.tool_dim}")

# Count parameters
total_params = sum(p.numel() for p in projector.parameters())
print(f"  - Total parameters: {total_params:,}")

# Test 3: Project Context Embeddings
print("\n" + "=" * 70)
print("Test 3: Project Context Embeddings")
print("=" * 70)

# Create dummy context embeddings
v_temporal = torch.randn(256)  # From Section 2.4
task_embedding = torch.randn(1, 10, 896)  # From Section 2.1 (batch=1, seq=10, dim=896)
tool_embeddings = torch.randn(1, 8, 1024)  # From Section 1.5 (batch=1, tools=8, dim=1024)

print(f"Input embeddings:")
print(f"  - v_temporal: {v_temporal.shape}")
print(f"  - task_embedding: {task_embedding.shape}")
print(f"  - tool_embeddings: {tool_embeddings.shape}")

projected = projector(
    v_temporal=v_temporal,
    task_embedding=task_embedding,
    tool_embeddings=tool_embeddings
)

print(f"\nProjected embeddings:")
for key, val in projected.items():
    print(f"  - {key}: {val.shape}")

assert 'temporal_tokens' in projected
assert 'task_tokens' in projected
assert 'tool_tokens' in projected

print(f"✓ Projection successful")

# Test 4: Prepare Inputs with Context
print("\n" + "=" * 70)
print("Test 4: Prepare LLM Inputs with Context")
print("=" * 70)

# Tokenize some text
prompt = "Select the best tool for this task:"
encoded = qwen.tokenizer(prompt, return_tensors='pt')
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']

print(f"Text prompt: '{prompt}'")
print(f"Input IDs shape: {input_ids.shape}")

# Prepare with context
inputs_embeds, new_attention_mask = qwen.prepare_inputs_with_context(
    input_ids,
    attention_mask,
    projected
)

print(f"\nWith context injection:")
print(f"  - Original length: {input_ids.size(1)}")
print(f"  - Context tokens: {inputs_embeds.size(1) - input_ids.size(1)}")
print(f"  - Total length: {inputs_embeds.size(1)}")
print(f"  - Embedding shape: {inputs_embeds.shape}")

# Breakdown
num_temporal = projected['temporal_tokens'].size(1)
num_task = projected['task_tokens'].size(1)
num_tools = projected['tool_tokens'].size(1)
num_text = input_ids.size(1)

print(f"\nToken breakdown:")
print(f"  - Temporal tokens: {num_temporal}")
print(f"  - Task tokens: {num_task}")
print(f"  - Tool tokens: {num_tools}")
print(f"  - Text tokens: {num_text}")
print(f"  - Total: {num_temporal + num_task + num_tools + num_text}")

print(f"✓ Context injection working")

# Test 5: Generate without Context (baseline)
print("\n" + "=" * 70)
print("Test 5: Generate Text (Baseline - No Context)")
print("=" * 70)

prompt = "What is the capital of France?"
print(f"Prompt: '{prompt}'")

qwen.model.eval()
with torch.no_grad():
    baseline_output = qwen.generate(
        input_text=prompt,
        context_embeddings=None,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=False  # Greedy for reproducibility
    )

print(f"Generated (baseline):")
print(f"  {baseline_output[0]}")
print(f"✓ Baseline generation working")

# Test 6: Generate with Context
print("\n" + "=" * 70)
print("Test 6: Generate Text with Context")
print("=" * 70)

prompt = "Based on the system resources and task requirements, which tool should be used?"
print(f"Prompt: '{prompt}'")

with torch.no_grad():
    context_output = qwen.generate(
        input_text=prompt,
        context_embeddings=projected,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=False
    )

print(f"Generated (with context):")
print(f"  {context_output[0]}")
print(f"✓ Context-aware generation working")

# Test 7: Complete TOMAS-LLM Model
print("\n" + "=" * 70)
print("Test 7: Complete TOMAS-LLM Model")
print("=" * 70)

from llm.model_wrapper import TOMASSLLMModel

print("Creating complete TOMAS-LLM model...")
tomas_model = TOMASSLLMModel.from_config(config)

print(f"✓ Model created")
print(f"Components loaded:")
print(f"  - Qwen backbone: {tomas_model.qwen is not None}")
print(f"  - Context projector: {tomas_model.projector is not None}")
print(f"  - Task encoder: {tomas_model.task_encoder is not None}")
print(f"  - Latency predictor: {tomas_model.latency_predictor is not None}")
print(f"  - Temporal encoder: {tomas_model.temporal_encoder is not None}")
print(f"  - Tool encoder: {tomas_model.tool_encoder is not None}")

# Test 8: Encode Context
print("\n" + "=" * 70)
print("Test 8: Encode Context (End-to-End)")
print("=" * 70)

user_task = "Process large image dataset with GPU acceleration"
print(f"User task: '{user_task}'")

with torch.no_grad():
    context = tomas_model.encode_context(
        user_task=user_task,
        predict_latency=True
    )

print(f"\nEncoded context:")
for key, val in context.items():
    if isinstance(val, torch.Tensor):
        print(f"  - {key}: shape {val.shape}")
    else:
        print(f"  - {key}: {val}")

print(f"✓ End-to-end context encoding working")

# Test 9: Full Generation Pipeline
print("\n" + "=" * 70)
print("Test 9: Full Generation Pipeline")
print("=" * 70)

prompt = "Recommend a tool for this task:"
print(f"Prompt: '{prompt}'")
print(f"Task: '{user_task}'")

tomas_model.eval()
with torch.no_grad():
    output = tomas_model.generate(
        prompt=prompt,
        user_task=user_task,
        predict_latency=True,
        max_new_tokens=50,
        do_sample=False
    )

print(f"\nGenerated recommendation:")
print(f"  {output[0]}")

print(f"✓ Full pipeline working")

# Test 10: Batch Generation
print("\n" + "=" * 70)
print("Test 10: Batch Generation")
print("=" * 70)

prompts = [
    "What tool for image processing?",
    "What tool for video conversion?",
    "What tool for text analysis?"
]

print(f"Batch prompts ({len(prompts)}):")
for i, p in enumerate(prompts):
    print(f"  {i+1}. {p}")

with torch.no_grad():
    batch_outputs = tomas_model.generate(
        prompt=prompts,
        user_task=user_task,
        predict_latency=True,
        max_new_tokens=30,
        do_sample=False
    )

print(f"\nBatch outputs:")
for i, out in enumerate(batch_outputs):
    print(f"  {i+1}. {out}")

print(f"✓ Batch generation working")

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)

print("\nSummary:")
print(f"  - Qwen2.5-0.5B loaded successfully")
print(f"  - Context projector: {total_params:,} parameters")
print(f"  - Context injection: temporal + task + tools → LLM")
print(f"  - Generation: baseline and context-aware modes")
print(f"  - Complete model: all components integrated")
print(f"  - End-to-end pipeline functional")
print(f"  ✓ Ready for training and inference!")
