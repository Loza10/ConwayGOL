## Zakkary Loveall
## ISYE 6644 Final Project
## Conway Game of Life w/ Metrics for stability analysis

import pygame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rows, cols = 40, 40
cell_size = 20
width, height = cols * cell_size, rows * cell_size

DEAD_COLOR = (30, 30, 30)
ALIVE_COLOR = (240, 240, 240)
GRID_COLOR = (50, 50, 50)

def displayBoard(screen, grid):
    screen.fill(GRID_COLOR)
    for i in range(rows):
        for j in range(cols):
            if (grid[i, j] == 1):
                color = ALIVE_COLOR
            else:
                color = DEAD_COLOR
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, color, rect)
    pygame.display.flip()

def countNeighbors(grid, row, col):
    tot = 0
    surround = [0, -1, 1]
    for i in surround:
        for j in surround:
            if i != 0 or j != 0:
                tot += grid[(row + i) % grid.shape[0], (col + j) % grid.shape[1]]
    return tot

def nextGen(grid):
    new_grid = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = countNeighbors(grid, i, j)
            if grid[i, j] == 1:
                if neighbors in [2, 3]:
                    new_grid[i, j] = 1
            else:
                if neighbors == 3:
                    new_grid[i, j] = 1
    return new_grid

def plotStatistics(data):

    sns.set(style="whitegrid")
    timesteps = data.shape[1]
    num_simulations = data.shape[0]

    plt.figure(figsize=(12, 6))
    sns.heatmap(data, cmap="viridis", cbar_kws={'label': 'Alive Cells'})
    plt.title("Alive Cells Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Simulation #")
    plt.tight_layout()
    plt.savefig("heatmap_alive_cells.png")
    plt.close()

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(means, label="Mean Alive Cells", color='blue')
    plt.fill_between(range(timesteps), means - stds, means + stds, color='blue', alpha=0.2, label="±1 Std Dev")
    plt.title("Mean Alive Cells Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Alive Cells")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mean_alive_cells_line.png")
    plt.close()

    final_alive = data[:, -1]
    plt.figure(figsize=(8, 5))
    sns.histplot(final_alive, bins=20, kde=True, color='purple')
    plt.title("Distribution of Final Alive Cell Counts")
    plt.xlabel("Alive Cells at Final Timestep")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("final_alive_histogram.png")
    plt.close()

    selected_steps = [0, timesteps // 2, timesteps - 1]
    data_for_box = [data[:, step] for step in selected_steps]

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data_for_box)
    plt.xticks([0, 1, 2], [f"Timestep {s}" for s in selected_steps])
    plt.title("Alive Cell Counts at Timesteps")
    plt.ylabel("Alive Cells")
    plt.tight_layout()
    plt.savefig("boxplot_selected_timesteps.png")
    plt.close()

    print("All plots saved.")


def main(num_simulations=100, steps=200):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CONWAY")

    clock = pygame.time.Clock()

    stats = []

    for sim in range(num_simulations):
        grid = np.random.choice([0, 1], size=(rows, cols), p=[0.8, 0.2])
        print(f"Running simulation {sim + 1} / {num_simulations}")

        alive_counts = []

        for step in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            displayBoard(screen, grid)
            alive_counts.append(np.sum(grid))
            grid = nextGen(grid)
            clock.tick(1000)
        
        stats.append(alive_counts)

        pygame.time.wait(500)

    stats = np.array(stats)
    plotStatistics(stats)
    pygame.quit()

if __name__ == "__main__":
    main()
