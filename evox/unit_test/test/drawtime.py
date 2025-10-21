import matplotlib.pyplot as plt
import numpy as np

# Updated data
population_size = np.array([100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
C_NSGA2_GPU = np.array([19.53404286, 57.71190643, 179.6190879, 523.6519832, 976.9816513, 1443.231255,
                        1923.236512, 2413.321486, 2898.694251, 3415.812238, 3847.366611])
PPS_GPU = np.array([20.54062643, 22.34901786, 30.08831542, 40.1595516, 91.5793321, 173.1971523,
                    191.0297651, 313.225684, 371.6523813, 452.5413568, 565.4152771])
CMOEA_MS_GPU = np.array([29.43405286, 102.49459, 328.7737267, 511.326871, 789.365125, 1012.987498,
                         1423.145983, 1768.549512, 2103.658279, 2423.895214, 2897.767394])
CCMO_GPU = np.array([25.62812286, 58.25938786, 360.7037429, 607.519621, 1213.985186, 1923.852413,
                     2543.325419, 3223.254183, 3812.752461, 4677.185423, 5394.011622])
EMCMO = np.array([40.31707857, 98.26650071, 415.4176758, 782.7515216, 1587.231549, 2466.365152,
                  3387.335412, 4256.223314, 5232.351642, 6335.623581, 7626.036629])
GMPEA_GPU = np.array([8.56812429, 8.683543571, 9.976735417, 10.13492235, 10.33782345, 10.87430231,
                      11.35023482, 12.87962344, 14.15292131, 16.65457852, 19.57103429])

# Plotting the curves
plt.figure(figsize=(6, 4))


# Define a function to plot each algorithm with specific styling
def plot_algorithm(y_data, label, color, gmpea_style=False):
    if gmpea_style:
        linewidth = 3  # Thicker line
        markersize = 8  # Larger marker
        alpha_fill = 0.25  # More prominent fill shade
    else:
        linewidth = 2  # Regular line thickness
        markersize = 5  # Regular marker size
        alpha_fill = 0.2  # Regular fill shade

    # Plot the average curve
    plt.plot(population_size, y_data,
             color=color, linewidth=linewidth,
             label=label, marker='o', markersize=markersize,
             markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)

    # Add fill between for GMPEA highlighting
    if gmpea_style:
        plt.fill_between(population_size, y_data - 0.5, y_data + 0.5, color=color, alpha=alpha_fill)


# Plot each algorithm
plot_algorithm(C_NSGA2_GPU, 'C-NSGAII-GPU', 'darkorange')
plot_algorithm(PPS_GPU, 'PPS-GPU ', 'forestgreen')
plot_algorithm(CMOEA_MS_GPU, 'CMOEA-MS-GPU ', 'steelblue')
plot_algorithm(CCMO_GPU, 'CCMO-GPU ', 'mediumvioletred')
plot_algorithm(EMCMO, 'EMCMO-GPU ', 'gold')
plot_algorithm(GMPEA_GPU, 'GMPEA-GPU', 'red', gmpea_style=True)

# Setting titles and labels
#plt.title('Average Time vs Population Size', fontsize=16)
plt.xlabel('Population Size', fontsize=14)
plt.ylabel('Average Time (seconds)', fontsize=14)

# Set x-axis and y-axis scales to logarithmic
#plt.xscale('log')
#plt.yscale('log')

# Set x-axis ticks
plt.xticks([100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])  # Display specific ticks

# Enable grid lines
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.7)

# Optimize legend display
legend = plt.legend(fontsize=10, loc='upper left', frameon=True, fancybox=True, shadow=True)
legend.get_frame().set_facecolor('#f0f0f0')
legend.get_frame().set_alpha(0.9)

# Highlight GMPEA in the legend
for text in legend.get_texts():
    if text.get_text() == 'GMPEA-GPU (This Study)':
        text.set_weight('bold')
        text.set_color('red')

plt.tight_layout()
plt.show()