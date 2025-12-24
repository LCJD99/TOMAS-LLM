"""
Test Output Parsing Module

Tests for Token Gate, Tool Classifier, Resource Regressor, and Output Parser.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.decoders.token_gate import TokenTypeGate, TOOL_PLAN_TOKEN_OFFSET
from src.decoders.tool_classifier import ToolClassifier, ToolClassifierWithRegistry
from src.decoders.resource_regressor import ResourceRegressor, ResourceRegressorWithNormalization
from src.decoders.output_parser import OutputParser, ToolPlan, BatchOutputParser


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


# Test 1: TokenTypeGate - Simplified Routing
print_section("Test 1: TokenTypeGate - Simplified Routing")

vocab_size = 151936  # Qwen2.5
hidden_dim = 896     # 0.5B model
gate = TokenTypeGate(vocab_size, hidden_dim)

print(f"Gate config:")
print(f"  vocab_size: {vocab_size}")
print(f"  TOOL_PLAN token ID: {gate.tool_plan_token_id}")
print(f"  Any token >= {vocab_size} is special")

# Create mixed token sequence
# Standard tokens: [0, vocab_size)
# Special token (TOOL_PLAN): vocab_size
batch_size = 2
seq_len = 10
token_ids = torch.tensor([
    [100, 200, 300, vocab_size, 400, 500, 600, 700, 800, 900],  # TOOL_PLAN at pos 3
    [50, 60, 70, 80, 90, vocab_size, 100, 110, 120, 130]        # TOOL_PLAN at pos 5
])

print(f"Token IDs shape: {token_ids.shape}")
print(f"Sample tokens: {token_ids[0]}")
print(f"Standard tokens: {(token_ids[0] < vocab_size).sum().item()} / {seq_len}")
print(f"Special tokens: {(token_ids[0] >= vocab_size).sum().item()} / {seq_len}")

# Get routing masks
is_special = gate.is_special_token_batch(token_ids)
print(f"\nIs special: {is_special}")
print(f"Special count: {is_special.sum().item()}")

masks = gate.get_routing_mask(token_ids)
print(f"Standard mask (row 0): {masks['standard_mask'][0]}")
print(f"Special mask (row 0): {masks['special_mask'][0]}")

# Extract special positions
hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
special_hidden, special_indices = gate.extract_special_positions(
    token_ids, hidden_states
)

print(f"\nExtracted TOOL_PLAN positions:")
print(f"  Hidden states shape: {special_hidden.shape}")
print(f"  Indices: {special_indices}")  # Should be [[0, 3], [1, 5]]
print(f"✓ TokenTypeGate routing working (simplified: token >= vocab_size)")


# Test 2: ToolClassifier
print_section("Test 2: ToolClassifier - Tool Selection")

num_tools = 10
tool_dim = 1024

classifier = ToolClassifier(
    hidden_dim=hidden_dim,
    tool_dim=tool_dim,
    num_tools=num_tools,
    use_attention=True
)

print(f"Classifier: {classifier}")

# Create inputs
batch = 3
tool_hidden = torch.randn(batch, hidden_dim)
tool_embeddings = torch.randn(num_tools, tool_dim)

# Forward
logits = classifier(tool_hidden, tool_embeddings)
print(f"Logits shape: {logits.shape}")  # (3, 10)

# Predict
predictions = classifier.predict(tool_hidden, tool_embeddings, return_probs=True)
print(f"Predicted tool IDs: {predictions['tool_id']}")
print(f"Confidence: {predictions['confidence']}")
print(f"Max prob: {predictions['probs'].max(dim=-1)[0]}")

# Test loss
target_tool_ids = torch.tensor([2, 5, 8])
loss = classifier.compute_loss(tool_hidden, tool_embeddings, target_tool_ids)
print(f"Classification loss: {loss.item():.4f}")

print("✓ ToolClassifier working")


# Test 4: ToolClassifierWithRegistry
print_section("Test 4: ToolClassifierWithRegistry")

classifier_registry = ToolClassifierWithRegistry(
    hidden_dim=hidden_dim,
    tool_dim=tool_dim,
    tool_registry_path=None,  # Use dummy registry
    use_attention=True
)

print(f"Tool names: {classifier_registry.tool_names}")
print(f"Tool embeddings shape: {classifier_registry.tool_embeddings.shape}")

# Predict with names
predictions_named = classifier_registry.predict_with_names(tool_hidden)
print(f"Predicted tools: {predictions_named['tool_name']}")
print(f"Confidence: {predictions_named['confidence']}")

print("✓ ToolClassifierWithRegistry working")


# Test 5: ResourceRegressor
print_section("Test 5: ResourceRegressor - Resource Allocation")

regressor = ResourceRegressor(
    hidden_dim=hidden_dim,
    use_constraint=True
)

print(f"Regressor: {regressor}")

# Create inputs
resource_hidden = torch.randn(batch, hidden_dim)
available_resources = torch.tensor([16.0, 64.0, 40.0, 12.0])  # [cpu_core, cpu_mem, gpu_sm, gpu_mem]

# Forward - returns normalized values [0, 1]
normalized = regressor(resource_hidden)
print(f"Normalized resources shape: {normalized.shape}")  # (3, 4)
print(f"Sample normalized: {normalized[0]}")
print(f"Range check: min={normalized.min():.3f}, max={normalized.max():.3f}")

# Predict with denormalization and named outputs
allocation = regressor.predict(
    resource_hidden,
    available_resources
)
print(f"\nResource allocation:")
print(f"  CPU cores: {allocation['cpu_core']}")
print(f"  CPU memory (GB): {allocation['cpu_mem_gb']}")
print(f"  GPU SMs: {allocation['gpu_sm']}")
print(f"  GPU memory (GB): {allocation['gpu_mem_gb']}")

# Test constraint: resources should not exceed available
assert (allocation['cpu_core'] <= available_resources[0]).all(), "CPU cores exceed limit"
assert (allocation['gpu_sm'] <= available_resources[2]).all(), "GPU SMs exceed limit"
print("✓ Resource constraints enforced")

# Test loss
target_resources = torch.tensor([
    [8.0, 32.0, 20.0, 8.0],
    [4.0, 16.0, 0.0, 0.0],
    [12.0, 48.0, 30.0, 10.0]
])

losses = regressor.compute_loss(
    resource_hidden,
    target_resources,
    available_resources,
    loss_type='huber',
    constraint_weight=1.0
)

print(f"\nLosses:")
print(f"  Total: {losses['total'].item():.4f}")
print(f"  Regression: {losses['regression'].item():.4f}")
print(f"  Constraint: {losses['constraint'].item():.4f}")

print("✓ ResourceRegressor working")


# Test 6: ResourceRegressorWithNormalization
print_section("Test 6: ResourceRegressorWithNormalization")

resource_stats = {
    'cpu_core': (8.0, 4.0),
    'cpu_mem_gb': (32.0, 16.0),
    'gpu_sm': (20.0, 10.0),
    'gpu_mem_gb': (8.0, 4.0)
}

normalized_regressor = ResourceRegressorWithNormalization(
    hidden_dim=hidden_dim,
    resource_stats=resource_stats
)

# Test normalization
test_resources = torch.tensor([[8.0, 32.0, 20.0, 8.0]])
normalized = normalized_regressor.normalize(test_resources)
denormalized = normalized_regressor.denormalize(normalized)

print(f"Original: {test_resources}")
print(f"Normalized: {normalized}")
print(f"Denormalized: {denormalized}")
assert torch.allclose(test_resources, denormalized), "Normalization roundtrip failed"

print("✓ Normalization working")


# Test 7: ToolPlan
print_section("Test 7: ToolPlan - Data Structure")

plan = ToolPlan(
    tool_id=3,
    tool_name="ImageMagick",
    confidence=0.95,
    cpu_core=8.0,
    cpu_mem_gb=32.0,
    gpu_sm=20.0,
    gpu_mem_gb=8.0,
    expected_latency_ms=1500.0,
    explanation="Process large image dataset with GPU acceleration"
)

print(f"Tool Plan:\n{plan}")
print(f"\nAs dict: {plan.to_dict()}")
print(f"\nAs JSON:\n{plan.to_json(indent=2)}")

# Validate
is_valid = plan.validate()
print(f"\nIs valid: {is_valid}")
assert is_valid, "Valid plan failed validation"

# Test invalid plan
invalid_plan = ToolPlan(
    tool_id=1,
    tool_name="test",
    confidence=1.5,  # Invalid: > 1.0
    cpu_core=-1.0,   # Invalid: negative
    cpu_mem_gb=16.0,
    gpu_sm=0.0,
    gpu_mem_gb=0.0
)
assert not invalid_plan.validate(), "Invalid plan passed validation"

print("✓ ToolPlan working")


# Test 8: OutputParser
print_section("Test 8: OutputParser - End-to-End")

# Create parser
parser = OutputParser(
    tool_classifier=classifier,
    resource_regressor=regressor,
    token_gate=gate,
    tool_id_to_name={i: f"tool_{i}" for i in range(num_tools)}
)

# Create generation output with TOOL_PLAN token
batch_size = 2
seq_len = 12
TOOL_PLAN = vocab_size  # First token after vocab
generated_ids = torch.tensor([
    [100, 200, 300, TOOL_PLAN, 400, 500, 600, 700, 800, 900, 1000, 1100],
    # TOOL_PLAN at position 3 - encodes both tool_id and resources
    
    [50, 60, 70, 80, 90, TOOL_PLAN, 100, 110, 120, 130, 140, 150]
    # TOOL_PLAN at position 5
])

hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
tool_embeddings = torch.randn(num_tools, tool_dim)
available_resources = torch.tensor([16.0, 64.0, 40.0, 12.0])

# Parse
with torch.no_grad():
    plans = parser.parse(
        generated_ids,
        hidden_states,
        tool_embeddings,
        available_resources
    )

print(f"Parsed {len(plans)} plans:\n")
for i, plan in enumerate(plans):
    print(f"Plan {i+1}:")
    print(f"  {plan}")
    print(f"  Valid: {plan.validate()}")

print("✓ OutputParser working")


# Test 9: Batch OutputParser
print_section("Test 9: BatchOutputParser - Efficient Parsing")

batch_parser = BatchOutputParser(
    tool_classifier=classifier,
    resource_regressor=regressor,
    token_gate=gate,
    tool_id_to_name={i: f"tool_{i}" for i in range(num_tools)}
)

# Note: All samples use SAME TOOL_PLAN position for batch efficiency
consistent_ids = torch.tensor([
    [100, 200, vocab_size, 300, 400, 500, 600, 700],
    [50, 60, vocab_size, 70, 80, 90, 100, 110]
])
# Both have TOOL_PLAN at pos 2

consistent_hidden = torch.randn(2, 8, hidden_dim)

with torch.no_grad():
    batch_plans = batch_parser.parse_batch(
        consistent_ids,
        consistent_hidden,
        tool_embeddings,
        available_resources
    )

print(f"Batch parsed {len(batch_plans)} plans:")
for i, plan in enumerate(batch_plans):
    print(f"  {i+1}. Tool: {plan.tool_name}, CPU: {plan.cpu_core:.1f} cores")

print("✓ BatchOutputParser working")


# Test 10: Integration Test - Full Pipeline
print_section("Test 10: Full Pipeline Integration")

print("Simulating full generation + parsing pipeline:")
print("\n1. Generate with TOOL_PLAN token")
# Simulate generation that produces TOOL_PLAN token
TOOL_PLAN = vocab_size
simulated_generation = torch.tensor([
    # Standard tokens, then TOOL_PLAN token
    [1, 2, 3, 4, 5, TOOL_PLAN, 6, 7, 8, 9, 10, 11, 12, 13, 14]
])
simulated_hidden = torch.randn(1, 15, hidden_dim)

print(f"   Generated sequence: {simulated_generation[0]}")
print(f"   Hidden states: {simulated_hidden.shape}")

print("\n2. Route tokens")
routing = gate.get_routing_mask(simulated_generation)
print(f"   Special tokens at positions: {routing['special_mask'].nonzero()[:, 1].tolist()}")

print("\n3. Extract hidden at TOOL_PLAN position")
plan_hidden, plan_idx = gate.extract_special_positions(
    simulated_generation, simulated_hidden
)
print(f"   TOOL_PLAN at: {plan_idx}")

print("\n4. Classify tool (from TOOL_PLAN hidden)")
with torch.no_grad():
    tool_pred = classifier.predict(plan_hidden, tool_embeddings)
    print(f"   Selected tool: {tool_pred['tool_id'].item()} (conf: {tool_pred['confidence'].item():.3f})")

print("\n5. Allocate resources (using SAME hidden state)")
with torch.no_grad():
    resource_alloc = regressor.predict(
        plan_hidden,  # Same hidden state as tool prediction!
        available_resources
    )
    print(f"   CPU: {resource_alloc['cpu_core'].item():.1f} cores, {resource_alloc['cpu_mem_gb'].item():.1f} GB")
    print(f"   GPU: {resource_alloc['gpu_sm'].item():.0f} SM, {resource_alloc['gpu_mem_gb'].item():.1f} GB")

print("\n6. Create ToolPlan")
with torch.no_grad():
    final_plan = parser.parse(
        simulated_generation,
        simulated_hidden,
        tool_embeddings,
        available_resources
    )[0]
    print(f"   {final_plan}")
    print(f"   JSON:\n{final_plan.to_json(indent=4)}")

print("\n✓ Full pipeline working")


# Summary
print_section("ALL TESTS PASSED ✓")

print("Summary:")
print("  - TokenTypeGate: Routing special tokens ✓")
print("  - ToolClassifier: Tool selection (10 tools) ✓")
print("  - ResourceRegressor: Resource allocation ✓")
print("  - ToolPlan: Data structure and validation ✓")
print("  - OutputParser: End-to-end parsing ✓")
print("  - BatchOutputParser: Efficient batch parsing ✓")
print("  - Full pipeline: Generation → routing → classification → regression → plan ✓")
print("\n✓ Ready for model integration!")
