"""
Lektion 1 - ML-ramverk och arkitektur
Assignment: Frameworks, tensors, and execution models

Instructions:
1. Complete the tasks below with short, runnable code snippets
2. Run each section and observe the output
3. Comment your code to explain what each part does
4. Keep everything in this file unless stated otherwise
"""

# Task 1: Vector and matrix basics (NumPy)
# TODO: Create two vectors (length 3) and compute:
# - dot product
# - L2 norm
# - cosine similarity
import numpy as np

# Både python-listnotation och numpyarray funkar
# Numpyarrayer är möjligtvis effektivare i minnet
a = [5 ,6 ,9]
b = [7, 21, 2]

vector1 = np.array([1, 2, 3])
vector2 = np.array([-487389247923874, 8, 13])

dot_prod = np.dot(vector1,vector2)
print("The dot product using two numpy arrays", dot_prod)
print("The dot product using a python list + python list", np.dot(a,b))
print("The dot product using a python list + numpy array", np.dot(a, vector1))

l2_norm = np.linalg.norm(vector1)
print("L2-normalisering för vector1 är:", l2_norm)

l2_norm_vector2 = np.linalg.norm(vector2)

cos_sim_v1_v2 = np.dot(vector1, vector2)/ (l2_norm * l2_norm_vector2)
cos_sim_v1_v2_ALTERNATIVE = dot_prod / (l2_norm * l2_norm_vector2)

print("The cosine similarity of vector1 and vector2 is: ", cos_sim_v1_v2)
print("The cosine similarity of vector1 and vector2 is: ", cos_sim_v1_v2_ALTERNATIVE)

# TODO: Create a 2x3 matrix and multiply it by a length-3 vector
matris1 = [[2, 4, 7], [1,5 ,6]]
matrix_multiplication = matris1 @ vector1
print(" Result of our matrix multiplication is:", matrix_multiplication)

# Task 2: Eager vs graph execution
# TODO: Write a small function f(x) = x^3 + 2x
def f(x):
    return x**3 + 2*x

print("Vi testar funktionen med 7, svaret är:", f(7))
# TODO: Implement f(x) in ONE of:
# - PyTorch (eager)
# - TensorFlow with @tf.function (graph)
# - JAX with @jit (graph-like)
import torch
x = torch.tensor(7.0)
y = f(x)
print(y)

# I regel är kör PyTorch kod i eager stil (en rad i taget)
# Om man använder torch.compile, så kör den i graph-stil
# jag tror dock att det inte funkar på MPS backend (Mac) och inte har CUDA
# TODO: Print the output and note how execution differs

# Med funktionen nedan kan vi undersöka om vi har cuda på vår dator
# Att byta till cuda/mps (alltså grafikkortet), ger oss många fler
# beräkningsenheter => mer effektiv beräkning
# Det här är en av de största anledningarna till PyTorch/TensorFlow
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# device for amd

print(device)

# Time är ett bibliotek, som låter oss mäta tid
# ofta använder vi det för att jämföra olika kod
import time

# 1. Eager Execution (Standard)
x = torch.randn(10000, device="cuda" if torch.cuda.is_available() else "cpu")
# Warm up
start_time = time.time()
_ = f(x)
eager_time = time.time() - start_time
print(f"Eager execution time: {eager_time} seconds")

# 2. Graph Execution (Compiled)
# torch.compile uses the inductor backend, which doesn't support MPS yet

# This 'traces' the function and optimizes the kernels
compiled_f = torch.compile(f)
# First call triggers the compilation (slower), subsequent calls are fast
start_time = time.time()
_ = compiled_f(x)
graph_time = time.time() - start_time
print(f"Graph execution time: {graph_time} seconds")


# Task 3: Framework comparison in code
# TODO: Using scikit-learn, load the iris dataset
# TODO: Train a LogisticRegression model
# TODO: Train a tiny MLP (MLPClassifier) on the same data
# TODO: Compare accuracy and write 3-5 comments in code about:
# - speed
# - API ergonomics
# - when you would pick each approach
