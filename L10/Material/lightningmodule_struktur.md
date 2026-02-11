# LightningModule Structure (Handout)

## Core methods

```python
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define layers and losses

    def forward(self, x):
        # inference logic
        return x

    def training_step(self, batch, batch_idx):
        # compute loss, log metrics
        return loss

    def validation_step(self, batch, batch_idx):
        # validation metrics
        return loss

    def configure_optimizers(self):
        # return optimizer(s) and schedulers
        return optimizer
```

## Typical data flow
- DataLoader yields batches
- `training_step` computes loss
- Lightning handles backward, optimizer step, and logging

## Tips
- Keep `forward` for inference only
- Log metrics with `self.log`
- Use callbacks for checkpoints and early stopping
