import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup (The "Brain" and the "Goal")
model = MyModel()
criterion = nn.MSELoss() # or CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. The Training Loop (The "Practice")
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # A. Reset: Clear previous gradients
        optimizer.zero_grad()
        
        # B. Forward: Build the graph & get prediction
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # C. Backward: AutoDiff calculates the "blame" (gradients)
        loss.backward()
        
        # D. Update: Optimizer moves weights down the hill
        optimizer.step()
        
    print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")