# DataLoader Performance Guide

## num_workers
- Start with `num_workers=0` to validate correctness
- Increase gradually (2, 4, 8) and measure throughput
- Too many workers can slow you down on small datasets

## pin_memory
- Use `pin_memory=True` when training on GPU
- Combine with `non_blocking=True` in `.to(device)`

## Example
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

for xb, yb in train_loader:
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
```

## Profiling tips
- Measure data loading time separately from compute
- Use PyTorch Profiler to locate bottlenecks
