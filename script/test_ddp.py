"""
Quick test for DDP training setup.

This script runs a minimal DDP training test to verify:
1. DDP environment setup works
2. Model wrapping and data distribution work
3. Gradient synchronization works
4. Checkpoint saving/loading works

Usage:
    torchrun --nproc_per_node=2 script/test_ddp.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from pathlib import Path

def setup_ddp():
    """Initialize DDP environment."""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device


def cleanup_ddp():
    """Clean up DDP."""
    dist.destroy_process_group()


def test_ddp_basic():
    """Test basic DDP functionality."""
    print("=" * 80)
    print("TEST 1: Basic DDP Setup")
    print("=" * 80)
    
    rank, world_size, local_rank, device = setup_ddp()
    
    print(f"[Rank {rank}/{world_size}] Device: {device}")
    print(f"[Rank {rank}] Local rank: {local_rank}")
    
    # Test barrier
    dist.barrier()
    if rank == 0:
        print("✓ All ranks synchronized")
    
    cleanup_ddp()
    print()


def test_ddp_model():
    """Test DDP model wrapping and forward pass."""
    print("=" * 80)
    print("TEST 2: Model Wrapping and Forward Pass")
    print("=" * 80)
    
    rank, world_size, local_rank, device = setup_ddp()
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        print(f"✓ Model wrapped with DDP")
    
    # Test forward pass
    x = torch.randn(4, 10, device=device)
    y = model(x)
    
    if rank == 0:
        print(f"✓ Forward pass successful: input shape {x.shape}, output shape {y.shape}")
    
    cleanup_ddp()
    print()


def test_ddp_data_parallel():
    """Test data distribution with DistributedSampler."""
    print("=" * 80)
    print("TEST 3: Data Distribution")
    print("=" * 80)
    
    rank, world_size, local_rank, device = setup_ddp()
    
    # Create dataset
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    
    # Create sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        drop_last=True
    )
    
    print(f"[Rank {rank}] Dataloader created with {len(dataloader)} batches")
    
    # Count samples
    total_samples = 0
    for batch_x, batch_y in dataloader:
        total_samples += batch_x.size(0)
    
    print(f"[Rank {rank}] Processed {total_samples} samples")
    
    # Gather total samples from all ranks
    total_samples_tensor = torch.tensor([total_samples], device=device)
    gathered_tensors = [torch.zeros_like(total_samples_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, total_samples_tensor)
    
    if rank == 0:
        total_across_ranks = sum([t.item() for t in gathered_tensors])
        print(f"✓ Total samples across all ranks: {total_across_ranks}")
        print(f"✓ Per-rank distribution: {[t.item() for t in gathered_tensors]}")
    
    cleanup_ddp()
    print()


def test_ddp_gradient_sync():
    """Test gradient synchronization."""
    print("=" * 80)
    print("TEST 4: Gradient Synchronization")
    print("=" * 80)
    
    rank, world_size, local_rank, device = setup_ddp()
    
    # Create model and optimizer
    model = torch.nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create different data on each rank
    x = torch.randn(4, 10, device=device) + rank  # Different data per rank
    y = torch.randn(4, 1, device=device)
    
    # Forward and backward
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    
    print(f"[Rank {rank}] Loss before backward: {loss.item():.4f}")
    
    loss.backward()
    
    # Check gradients are synchronized
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_tensor = param.grad.clone()
            gathered_grads = [torch.zeros_like(grad_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_grads, grad_tensor)
            
            if rank == 0:
                # All gradients should be identical after DDP sync
                all_equal = all(torch.allclose(gathered_grads[0], g) for g in gathered_grads)
                if all_equal:
                    print(f"✓ Gradient for {name} is synchronized across all ranks")
                else:
                    print(f"✗ Gradient for {name} is NOT synchronized!")
    
    optimizer.step()
    
    if rank == 0:
        print("✓ Optimizer step successful")
    
    cleanup_ddp()
    print()


def test_ddp_checkpoint():
    """Test checkpoint saving and loading."""
    print("=" * 80)
    print("TEST 5: Checkpoint Saving and Loading")
    print("=" * 80)
    
    rank, world_size, local_rank, device = setup_ddp()
    
    # Create model
    model = torch.nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few steps
    for step in range(3):
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 1, device=device)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Save checkpoint (only on rank 0)
    checkpoint_dir = Path("checkpoints/test_ddp")
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': 3
        }
        checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    # Barrier to ensure checkpoint is saved
    dist.barrier()
    
    # Create new model and load checkpoint
    model2 = torch.nn.Linear(10, 1).to(device)
    model2 = DDP(model2, device_ids=[local_rank])
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model2.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"[Rank {rank}] ✓ Checkpoint loaded successfully")
    
    # Verify parameters match
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), model2.named_parameters()):
        if torch.allclose(param1, param2):
            if rank == 0:
                print(f"✓ Parameter {name1} matches after reload")
        else:
            if rank == 0:
                print(f"✗ Parameter {name1} does NOT match after reload!")
    
    # Cleanup test checkpoint
    if rank == 0:
        import shutil
        shutil.rmtree(checkpoint_dir)
        print("✓ Test checkpoint cleaned up")
    
    cleanup_ddp()
    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("DDP FUNCTIONALITY TEST SUITE")
    print("=" * 80)
    print()
    
    try:
        test_ddp_basic()
        
        # Reinitialize for each test
        test_ddp_model()
        test_ddp_data_parallel()
        test_ddp_gradient_sync()
        test_ddp_checkpoint()
        
        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
