import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class SpringVizualizer:
    fig, ax = plt.subplots(1, 1)
    sim_len = None

    def animate(self, i, x, u, x_lim, u_lim):
        spacing = 2 * x_lim
        self.ax.clear()
        for j in range(int(x.shape[0] / 2)):
            self.ax.plot(
                [spacing * j, spacing * j], [0.1, -0.1], color="k", linestyle="--"
            )
            self.ax.plot(
                [spacing * j - x_lim, spacing * j - x_lim],
                [0.1, -0.1],
                color="r",
                linestyle="--",
            )
            self.ax.plot(
                [spacing * j + x_lim, spacing * j + x_lim],
                [0.1, -0.1],
                color="r",
                linestyle="--",
            )
            self.ax.plot(
                x[2 * j, i] + spacing * j, 0, color="r", marker="o", markersize=10
            )
            mag = 1  # Scaling input for plotting
            color = "b"
            if np.abs(u[j, i]) >= u_lim:
                color = "r"
            if u[j, i] != 0:
                self.ax.arrow(
                    x[2 * j, i] + spacing * j,
                    0,
                    mag * u[j, i],
                    0,
                    color=color,
                    head_width=0.03,
                    head_length=0.2,
                )
                # ax.arrow(spacing*j, 0.1, mag*u[j, i], 0, color = 'b', head_width=0.03, head_length=0.2)

        self.ax.set_xlim([-x_lim - 1, spacing * (int(x.shape[0] / 2) - 1) + x_lim + 1])
        self.ax.set_ylim([-1, 1])
        self.ax.axis("off")

        plt.savefig(
            f"C:/Users/shmallick/Github/tmp/img/img_{i}.png",
            transparent=False,
            facecolor="white",
        )

    def spring_sys_viz(self, x, u, rep=False):
        self.sim_len = x.shape[1] - 1

        ani = FuncAnimation(
            self.fig,
            lambda i: self.animate(i, x, u, x_lim=5, u_lim=20),
            frames=x.shape[1] - 1,
            interval=1,
            repeat=rep,
        )
        plt.show()

    def create_gif(self, name):
        if self.sim_len is None:
            raise RuntimeError("must run spring_sys_viz before trying to create gif.")
        frames = []
        for i in range(int(self.sim_len)):
            image = imageio.v2.imread(f"C:/Users/shmallick/Github/tmp/img/img_{i}.png")
            frames.append(image)
            imageio.mimsave(
                "C:/Users/shmallick/Github/tmp/gifs/"
                + str(name)
                + ".gif",  # output gif
                frames,  # array of input frames
                fps=5,
            )
