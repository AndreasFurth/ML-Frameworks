import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup (The "Brain" and the "Goal")
# Vi behöver alltid ha en modell. Vi definerar typiskt
# många av dess delar (input_size, hidden layers 
# (antal noder OCH antal lager), output size)
# Ibland använder vi andras modeller. 
model = MyModel()

# Sedan definerar vi hur vår loss ska räknas ut
# Typiskt CrossEntropyLoss för klassifikation
# Mean square error för regression
# Det finns många andra
# Loss är alltså: Hur fel har vår modell (mängd error)
# Loss är det som vår modell försöker minimera
criterion = nn.CrossEntropyLoss() # or MSELoss for regression

# Nästa steg är typiskt att definera en optimizer. 
# Alltså: en modul som kan optimera vår lossfunktion åt oss
# BASELINE: är typiskt Stochastic Gradient Descent
# MEN: Vi börjar oftast med ADAM ändå, det är en avancerad SGD
# Learning Rate är en av, kanske den VIKTIGASTE hyperparametern
# En för stor kan leda till att vi inte hittar optimum, en för liten
# kan riskera lokala minimum, eller att vi inte konvergerar alls
# OFTA är typiska värden någon negativ tiopotens, alltså (0.1, 0.01, 0.001 osv)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate är något som kan ha stor inverkan på resultat
# Det är ofta något vi experimenterar med. Det finns specifika 
# bibliotek som hjälper oss att optimera t.ex learning rate
# MEN: vi kan göra en egen loop, som ned 
# Optuna t.ex
# lr_list = [0.1, 0.01, 0.001]

# for lr in lr_list: 
    

# 2. The Training Loop (The "Practice")
# Traning-loopen syftar till att träna vårt nätverk (vår modell)
# på vår data. Det gör vi genom att skicka in datan hinkvis med gånger
# och jämföra resultaten med facit (= räkna ut loss), 
# och efterhand justera vikterna. 

# vi gör den här processen i epocher, alltså ett visst antal gånger
# (vi bestämmer hur många)
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # A. Reset: Clear previous gradients
        # Vi säger till optimizern att glömma vad den gjorde förra varvet
        # (det kan gå snett annars)
        optimizer.zero_grad()
        
        # B. Forward: Build the graph & get prediction
        # Vi stoppar in vår data in i modellen, och tar fram några outputs
        # (datan går en vända framåt i modellen)
        outputs = model(inputs)

        # Vi stoppar in våra outputs och targets i vår Loss-modell
        # för att räkna ut vår lossfunktion (hur mycket fel vår modell hade)
        # Vi jämför alltså modellens predictions med y (target)
        loss = criterion(outputs, targets)
        
        # C. Backward: AutoDiff calculates the "blame" (gradients)
        # Vi börjar med att räkna ut den justering av vikter, som 
        # leder till störst minskning av Loss.
        loss.backward()
        
        
        # D. Update: Optimizer moves weights down the hill
        # Sen uppdaterar vi vikterna (hoppar i lutningens riktning)
        optimizer.step()
        
        # I regel vill vi spara modellen. Ofta sparar vi även
        # Checkpoint under tiden vi tränar (alltså, modellen 
        # efter ett visst antal epocher.) Om lagring inte är en faktor
        # bör man spara varje epoch. 
        
    print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")