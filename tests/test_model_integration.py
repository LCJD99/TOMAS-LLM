"""
Test TOMAS-LLM Model Integration with Output Heads

Tests the complete model with all components:
- LLM backbone (Qwen2.5)
- Context projector
- Output heads (TokenTypeGate, ToolClassifier, ResourceRegressor, OutputParser)
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm.qwen_backbone import QwenBackbone, ContextProjector
from src.llm.model_wrapper import TOMASSLLMModel
from src.decoders import (
    TokenTypeGate,
    ToolClassifier,
    ResourceRegressor,
    OutputParser,
    TOOL_PLAN_TOKEN_OFFSET
)


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


# Test 1: Create Model Components
print_section("Test 1: Create Model Components")

print("1. Creating Qwen backbone (0.5B test model)...")
qwen = QwenBackbone(model_name="Qwen/Qwen2.5-0.5B-Instruct")
print(f"   Vocab size: {qwen.model.config.vocab_size}")
print(f"   Hidden dim: {qwen.model.config.hidden_size}")
print(f"   Num layers: {qwen.model.config.num_hidden_layers}")

vocab_size = qwen.model.config.vocab_size
hidden_dim = qwen.model.config.hidden_size

print("\n2. Creating context projector...")
projector = ContextProjector(
    llm_hidden_dim=hidden_dim,
    temporal_dim=256,
    task_dim=768,
    tool_dim=1024
)
print(f"   Projector: {projector}")

print("\n3. Creating output heads...")
# TokenTypeGate
token_gate = TokenTypeGate(
    vocab_size=vocab_size,
    hidden_dim=hidden_dim
)
print(f"   Token gate: vocab_size={vocab_size}")
print(f"   TOOL_PLAN token ID: {token_gate.tool_plan_token_id}")

# ToolClassifier
num_tools = 5
tool_dim = 1024
tool_classifier = ToolClassifier(
    hidden_dim=hidden_dim,
    tool_dim=tool_dim,
    num_tools=num_tools,
    use_attention=True
)
print(f"   Tool classifier: {num_tools} tools")

# ResourceRegressor
resource_regressor = ResourceRegressor(
    hidden_dim=hidden_dim,
    use_constraint=True
)
print(f"   Resource regressor: output_dim=4 (normalized)")

# OutputParser
tool_id_to_name = {
    0: "ImageMagick",
    1: "FFmpeg",
    2: "Blender",
    3: "TensorFlow",
    4: "PyTorch"
}

output_parser = OutputParser(
    tool_classifier=tool_classifier,
    resource_regressor=resource_regressor,
    token_gate=token_gate,
    tool_id_to_name=tool_id_to_name
)
print(f"   Output parser created")

print("\n✓ All components created")


# Test 2: Create Complete Model
print_section("Test 2: Create Complete TOMAS-LLM Model")

model = TOMASSLLMModel(
    qwen_backbone=qwen,
    context_projector=projector,
    token_gate=token_gate,
    tool_classifier=tool_classifier,
    resource_regressor=resource_regressor,
    output_parser=output_parser,
    tool_id_to_name=tool_id_to_name
)

print("Model components:")
print(f"  - LLM backbone: {type(model.qwen).__name__}")
print(f"  - Context projector: {type(model.projector).__name__}")
print(f"  - Token gate: {type(model.token_gate).__name__}")
print(f"  - Tool classifier: {type(model.tool_classifier).__name__}")
print(f"  - Resource regressor: {type(model.resource_regressor).__name__}")
print(f"  - Output parser: {type(model.output_parser).__name__}")

print("\n✓ Complete model created")


# Test 3: Forward Pass
print_section("Test 3: Forward Pass with Output Heads")

# Create dummy input
batch_size = 2
seq_len = 20
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

print(f"Input shape: {input_ids.shape}")

# Forward pass with hidden states
print("\nForward pass...")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )

print(f"Output keys: {outputs.keys()}")
print(f"Logits shape: {outputs.logits.shape}")
if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
    print(f"Hidden states: {len(outputs.hidden_states)} layers")
    print(f"Last layer hidden: {outputs.hidden_states[-1].shape}")

print("\n✓ Forward pass successful")


# Test 4: Parse Output
print_section("Test 4: Parse Generated Output")

# Simulate generation output with TOOL_PLAN token
TOOL_PLAN = vocab_size + TOOL_PLAN_TOKEN_OFFSET
generated_ids = torch.tensor([
    [100, 200, 300, TOOL_PLAN, 400, 500, 600, 700],
    [50, 60, TOOL_PLAN, 70, 80, 90, 100, 110]
])

# Create corresponding hidden states
gen_seq_len = generated_ids.size(1)
hidden_states = torch.randn(batch_size, gen_seq_len, hidden_dim)

# Create tool embeddings
tool_embeddings = torch.randn(num_tools, tool_dim)

# Available resources
available_resources = torch.tensor([16.0, 64.0, 40.0, 12.0])  # [cpu, mem, gpu_sm, gpu_mem]

print(f"Generated IDs shape: {generated_ids.shape}")
print(f"TOOL_PLAN positions:")
for i in range(batch_size):
    plan_pos = (generated_ids[i] == TOOL_PLAN).nonzero(as_tuple=True)[0]
    if len(plan_pos) > 0:
        print(f"  Sample {i}: position {plan_pos[0].item()}")

# Parse output
print("\nParsing output...")
with torch.no_grad():
    tool_plans = model.parse_output(
        generated_ids=generated_ids,
        hidden_states=hidden_states,
        tool_embeddings=tool_embeddings,
        available_resources=available_resources
    )

print(f"\nParsed {len(tool_plans)} plans:")
for i, plan in enumerate(tool_plans):
    print(f"\nPlan {i+1}:")
    print(f"  Tool: {plan.tool_name} (ID: {plan.tool_id}, conf: {plan.confidence:.3f})")
    print(f"  CPU: {plan.cpu_core:.0f} cores, {plan.cpu_mem_gb:.1f} GB")
    print(f"  GPU: {plan.gpu_sm:.0f} SM, {plan.gpu_mem_gb:.1f} GB")
    print(f"  Valid: {plan.validate()}")

print("\n✓ Output parsing successful")


# Test 5: From Config
print_section("Test 5: Create Model from Config")

config = {
    'llm': {
        'model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
        'device_map': 'cpu',
        'torch_dtype': 'float32'
    },
    'context': {
        'temporal_dim': 256,
        'task_dim': 768,
        'tool_dim': 1024
    },
    'decoder': {
        'enabled': True,
        'tool_dim': 1024,
        'num_tools': 5,
        'use_attention': True,
        'use_constraint': True,
        'tool_id_to_name': {
            0: "ImageMagick",
            1: "FFmpeg",
            2: "Blender",
            3: "TensorFlow",
            4: "PyTorch"
        }
    }
}

print("Creating model from config...")
model_from_config = TOMASSLLMModel.from_config(config)

print(f"\nModel components from config:")
print(f"  - LLM: {model_from_config.qwen is not None}")
print(f"  - Projector: {model_from_config.projector is not None}")
print(f"  - Token gate: {model_from_config.token_gate is not None}")
print(f"  - Tool classifier: {model_from_config.tool_classifier is not None}")
print(f"  - Resource regressor: {model_from_config.resource_regressor is not None}")
print(f"  - Output parser: {model_from_config.output_parser is not None}")

if model_from_config.output_parser is not None:
    print(f"\nTool registry: {model_from_config.output_parser.tool_id_to_name}")

print("\n✓ Model from config successful")


# Summary
print_section("ALL TESTS PASSED ✓")

print("Summary:")
print("  - Model components creation ✓")
print("  - Complete model assembly ✓")
print("  - Forward pass with hidden states ✓")
print("  - Output parsing into ToolPlan ✓")
print("  - Config-based initialization ✓")
print("\n✓ TOMAS-LLM model integration complete!")
print("\nNext steps:")
print("  - Add generation with parsing test")
print("  - Integrate with training loop")
print("  - Add end-to-end inference pipeline")
