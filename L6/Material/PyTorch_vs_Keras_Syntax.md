# PyTorch vs. Keras Syntax (Quick Guide)

This guide highlights common syntax patterns side by side. It is not exhaustive.

## Tensor creation

**PyTorch**
```python
import torch
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.zeros((2, 3))
```

**Keras / TensorFlow**
```python
import tensorflow as tf
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
y = tf.zeros((2, 3))
```

## Model definition

**PyTorch**
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)
```

**Keras**
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(10,)),
    keras.layers.Dense(1),
])
```

## Training loop vs. compile/fit

**PyTorch**
```python
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for xb, yb in dataloader:
    optimizer.zero_grad()
    pred = model(xb)
    loss = loss_fn(pred, yb)
    loss.backward()
    optimizer.step()
```

**Keras**
```python
model.compile(optimizer="adam", loss="mse")
model.fit(dataset, epochs=5)
```

## Evaluation

**PyTorch**
```python
model.eval()
with torch.no_grad():
    preds = model(x_test)
```

**Keras**
```python
model.evaluate(x_test, y_test)
```

## Saving models

**PyTorch**
```python
torch.save(model.state_dict(), "model.pt")
```

**Keras**
```python
model.save("model.keras")
```
