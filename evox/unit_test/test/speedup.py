import matplotlib.pyplot as plt
import numpy as np

# Simulated data
algorithms = ['c-NSGA-II', 'PPS', 'CMOEA-MS', 'CCMO', 'EMCMO', 'GMPEA']
cpu_times = [1223.23, 321.79, 1678.3, 1810.5, 2163.1, 623.8]  # CPU runtime (seconds)
gpu_times = [191.45, 20.75, 328.77, 360.70, 415.41, 9.97]       # GPU runtime (seconds)

# Calculate speedup
speedups = [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(6, 4))

# Set bar positions
x = np.arange(len(algorithms))
width = 0.35

# Draw grouped bar chart for CPU and GPU times
bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU Time', color='skyblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU Time', color='lightcoral', alpha=0.8)

# Set left y-axis (time)
ax1.set_ylabel('Runtime (seconds)', fontsize=14)
#ax1.set_title('CPU and GPU Runtime Comparison and Speedup for Different Algorithms', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
# 第一个图例在最上方
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax1.grid(True, alpha=0.3)

# Create right y-axis for speedup
ax2 = ax1.twinx()
line = ax2.plot(x, speedups, color='green', marker='o', linewidth=2, markersize=6, label='Speedup')
ax2.set_ylabel('Speedup (GPU time / CPU time)', fontsize=14, color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Add value labels on bar charts
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Add value labels on speedup line
for i, (xi, yi) in enumerate(zip(x, speedups)):
    ax2.annotate(f'{yi:.1f}x',
                xy=(xi, yi),
                xytext=(0, 10),  # 10 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, color='green', fontweight='bold')

# 第二个图例稍微向下一点
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.8))


# 指定左侧纵轴显示范围（运行时间）
ax1.set_ylim(0, 3000)  # 范围：0-130秒

# 指定右侧纵轴显示范围（加速比）
ax2.set_ylim(0, 280)  # 范围：0-12倍
# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Optional: save image
# plt.savefig('algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')

# Print simulated data for reference
print("Simulated Data:")
print("Algorithms:", algorithms)
print("CPU Times:", cpu_times)
print("GPU Times:", gpu_times)
print("Speedups:", [f"{s:.1f}x" for s in speedups])