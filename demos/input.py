#!/home/dawnat9/miniconda3/envs/llm/bin/python
"""
Demo: Complete Center Panel Pipeline (Sections 2.1-2.4)

This script demonstrates the full runtime context encoding pipeline:
1. User Task Embedding (Section 2.1)
2. Latency Prediction (Section 2.2)
3. System Timeline Snapshot (Section 2.3)
4. Temporal Encoding (Section 2.4)

All components integrated and ready for LLM backbone (Section 3.x)
"""

# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import yaml

from src.context.user_task import UserTaskEncoder
from src.context.latency_predictor import LatencyPredictor
from src.context.timeline import SystemTimeline, ResourcePredictor
from src.context.temporal_encoder import TemporalEncoder

print("=" * 70)
print("TOMAS-LLM Center Panel Demo - Sections 2.1-2.4")
print("=" * 70)

# Load config
with open('configs/simple-test.yaml') as f:
    config = yaml.safe_load(f)

print("\n1. User Task Embedding (Section 2.1)")
print("-" * 70)

user_task = "Process this image dataset using GPU acceleration for faster inference"
print(f"Task: {user_task}")

task_encoder = UserTaskEncoder.from_config(config)
task_output = task_encoder(user_task)
# UserTaskEncoder returns (pooled_embedding, sequence_embeddings)
task_embedding = task_output[0] if isinstance(task_output, tuple) else task_output

print(f"Task embedding shape: {task_embedding.shape}")
print(f"Task embedding norm: {task_embedding.norm():.4f}")
print("✓ Task encoded")

print("\n2. Latency Prediction (Section 2.2)")
print("-" * 70)

latency_predictor = LatencyPredictor(mode='fixed', fixed_latency_ms=1000)
t_inf = latency_predictor()

print(f"Predicted latency (T_inf): {t_inf.item():.0f}ms")
print("✓ Latency predicted")

print("\n3. System Timeline Snapshot (Section 2.3)")
print("-" * 70)

timeline = SystemTimeline(
    csv_path='input/system_profiling.csv',
    interpolation='linear'
)

resource_predictor = ResourcePredictor(timeline=timeline)
resource_snapshot = resource_predictor(t_inf, return_dict=True)

print(f"Resource snapshot at T_inf={t_inf.item():.0f}ms:")
print(f"  CPU cores: {resource_snapshot['cpu_cores']:.1f}")
print(f"  CPU memory: {resource_snapshot['cpu_mem_gb']:.1f} GB")
print(f"  GPU SM: {resource_snapshot['gpu_sm']:.1f}%")
print(f"  GPU memory: {resource_snapshot['gpu_mem_gb']:.1f} GB")
print("✓ Resource snapshot retrieved")

print("\n4. Temporal Encoding (Section 2.4)")
print("-" * 70)

temporal_encoder = TemporalEncoder.from_config(config)
v_temporal = temporal_encoder(t_inf)

print(f"Temporal embedding shape: {v_temporal.shape}")
print(f"Temporal embedding norm: {v_temporal.norm():.4f}")
print(f"Temporal embedding sample: {v_temporal[:5]}")
print("✓ Temporal features encoded")

print("\n" + "=" * 70)
print("Complete Pipeline Summary")
print("=" * 70)

print(f"\nInputs:")
print(f"  - User task: '{user_task}'")
print(f"  - System timeline: input/system_profiling.csv (31 snapshots)")

print(f"\nOutputs (ready for LLM injection):")
print(f"  - Task embedding: shape {task_embedding.shape}, norm {task_embedding.norm():.4f}")
print(f"  - Predicted T_inf: {t_inf.item():.0f}ms")
print(f"  - Resource snapshot: CPU={resource_snapshot['cpu_cores']:.0f}, GPU={resource_snapshot['gpu_sm']:.0f}%")
print(f"  - Temporal embedding: shape {v_temporal.shape}, norm {v_temporal.norm():.4f}")

print(f"\nNext Steps (Section 3.x):")
print(f"  1. Project v_temporal to LLM dimension (256 → 3584)")
print(f"  2. Concatenate with task_embedding and tool encodings")
print(f"  3. Feed to Qwen2.5-7B LLM backbone")
print(f"  4. Generate tool selection and execution plan")

print("\n" + "=" * 70)
print("Center Panel Ready! ✓")
print("=" * 70)

# Batch processing demo
print("\n" + "=" * 70)
print("Bonus: Batch Processing Demo")
print("=" * 70)

# Multiple tasks
tasks = [
    "Resize images to 224x224 for model training",
    "Convert video to different formats",
    "Run object detection on image dataset"
]

print(f"\nProcessing {len(tasks)} tasks in batch:")
for i, task in enumerate(tasks):
    print(f"  {i+1}. {task}")

# Encode all tasks
task_outputs = [task_encoder(task) for task in tasks]
task_embeddings = [out[0] if isinstance(out, tuple) else out for out in task_outputs]

print(f"\nTask embeddings (variable length sequences):")
for i, emb in enumerate(task_embeddings):
    print(f"  Task {i+1}: shape {emb.shape}, norm {emb.norm():.4f}")

# Different latencies for different tasks
t_inf_batch = torch.tensor([500.0, 1000.0, 1500.0])
print(f"Latency batch: {t_inf_batch.numpy()} ms")

# Temporal encoding for each latency
v_temporal_batch = temporal_encoder(t_inf_batch)
print(f"Temporal batch shape: {v_temporal_batch.shape}")

# Resource snapshots
print(f"\nResource snapshots:")
for i, t in enumerate(t_inf_batch):
    snapshot = resource_predictor(t, return_dict=True)
    print(f"  T={t:.0f}ms: CPU={snapshot['cpu_cores']:.0f}, GPU={snapshot['gpu_sm']:.0f}%")

print("\n✓ Batch processing complete!")

print("\n" + "=" * 70)
print("All Systems Operational!")
print("=" * 70)
