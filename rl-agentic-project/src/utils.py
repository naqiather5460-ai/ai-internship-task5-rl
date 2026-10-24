import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(x, w=100):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(rewards, epsilons, title, path, w=100):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rewards, alpha=0.25, label="episode reward")
    ax[0].plot(np.arange(w - 1, len(rewards)), moving_average(rewards, w),
               label=f"moving avg ({w})")
    ax[0].set(title=f"{title}: reward", xlabel="episode", ylabel="total reward")
    ax[0].legend()
    ax[1].plot(epsilons)
    ax[1].set(title="Exploration rate (epsilon)", xlabel="episode", ylabel="epsilon")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
