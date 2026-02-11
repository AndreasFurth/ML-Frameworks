"""
Visualize optimization paths for different optimizers on a simple 2D loss surface.

Run:
  python optimization_paths_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt


def loss_fn(x, y):
    # A non-convex surface with multiple valleys.
    return 0.2 * (x ** 2 + y ** 2) + np.sin(1.5 * x) * np.cos(1.5 * y)


def grad_fn(x, y):
    d_dx = 0.4 * x + 1.5 * np.cos(1.5 * x) * np.cos(1.5 * y)
    d_dy = 0.4 * y - 1.5 * np.sin(1.5 * x) * np.sin(1.5 * y)
    return d_dx, d_dy


def run_optimizer(name, lr=0.1, steps=50, momentum=0.9, beta1=0.9, beta2=0.999):
    x, y = 2.5, -2.0
    vx, vy = 0.0, 0.0
    mx, my = 0.0, 0.0
    vx2, vy2 = 0.0, 0.0
    path = [(x, y)]

    for t in range(1, steps + 1):
        gx, gy = grad_fn(x, y)

        if name == "sgd":
            x -= lr * gx
            y -= lr * gy
        elif name == "momentum":
            vx = momentum * vx + lr * gx
            vy = momentum * vy + lr * gy
            x -= vx
            y -= vy
        elif name == "adam":
            mx = beta1 * mx + (1 - beta1) * gx
            my = beta1 * my + (1 - beta1) * gy
            vx2 = beta2 * vx2 + (1 - beta2) * (gx ** 2)
            vy2 = beta2 * vy2 + (1 - beta2) * (gy ** 2)
            mx_hat = mx / (1 - beta1 ** t)
            my_hat = my / (1 - beta1 ** t)
            vx_hat = vx2 / (1 - beta2 ** t)
            vy_hat = vy2 / (1 - beta2 ** t)
            x -= lr * mx_hat / (np.sqrt(vx_hat) + 1e-8)
            y -= lr * my_hat / (np.sqrt(vy_hat) + 1e-8)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

        path.append((x, y))

    return np.array(path)


def main():
    grid = np.linspace(-3.5, 3.5, 200)
    X, Y = np.meshgrid(grid, grid)
    Z = loss_fn(X, Y)

    paths = {
        "sgd": run_optimizer("sgd", lr=0.08),
        "momentum": run_optimizer("momentum", lr=0.08),
        "adam": run_optimizer("adam", lr=0.08),
    }

    plt.figure(figsize=(7, 6))
    plt.contour(X, Y, Z, levels=40, cmap="viridis")
    for name, path in paths.items():
        plt.plot(path[:, 0], path[:, 1], marker="o", markersize=2, label=name)
    plt.title("Optimization Paths")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
