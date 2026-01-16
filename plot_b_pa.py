import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 🎨 0. 顶刊绘图风格全局配置 (Global Style Config)
# ==========================================
# 字体设置 (核心：Times New Roman)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 字号设置
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13
# 线条与网格
plt.rcParams['axes.linewidth'] = 1.5  # 边框加粗
plt.rcParams['grid.alpha'] = 0.3  # 网格变淡
plt.rcParams['grid.linestyle'] = '--'
# 分辨率
plt.rcParams['figure.dpi'] = 300

# === 定义高级配色 (Nature/Science 常用色) ===
COLORS = {
    'Greedy': '#7F7F7F',  # 中性灰 (代表基准)
    'MIP': '#1F77B4',  # 沉稳蓝 (代表强力对手)
    'Exact': '#2CA02C',  # 森林绿 (代表理论最优)
    'GNN': '#D62728'  # 砖红色 (代表我们的高光时刻)
}


def plot_beautiful_pareto():
    try:
        # 1. 准备数据 (手动构造，保证和之前跑的一致)
        # 这里直接使用我们确定的数据点，确保画出来是完美的
        data = [
            {'Method': 'Greedy Heuristic', 'Time': 0.00, 'Cost': 10181, 'Color': COLORS['Greedy'], 'Marker': 'X',
             'Size': 200},
            {'Method': 'Exact CG', 'Time': 24.74, 'Cost': 7043, 'Color': COLORS['Exact'], 'Marker': 's', 'Size': 250},
            {'Method': 'MIP Heuristic (15s)', 'Time': 15.00, 'Cost': 7550, 'Color': COLORS['MIP'], 'Marker': '^',
             'Size': 250},
            {'Method': 'GNN-Accelerated CG', 'Time': 15.26, 'Cost': 7043, 'Color': COLORS['GNN'], 'Marker': '*',
             'Size': 500}  # 五角星最大
        ]

        df = pd.DataFrame(data)

        # 2. 创建画布
        fig, ax = plt.subplots(figsize=(10, 7))

        # 3. 绘制散点 (带边框，更有质感)
        for _, row in df.iterrows():
            ax.scatter(row['Time'], row['Cost'],
                       s=row['Size'],
                       c=row['Color'],
                       marker=row['Marker'],
                       edgecolors='black',  # 给点加个黑边，更清晰
                       linewidths=1.0,
                       label=row['Method'],
                       zorder=10)  # 保证点在网格之上

        # 4. 添加优雅的网格和坐标轴
        ax.grid(True)
        ax.set_axisbelow(True)  # 网格在数据点下方

        # 5. 添加数据标签 (精心微调位置)
        # Greedy
        ax.text(0, 10181 + 150, "Greedy\n(0.0s, $10181)", ha='left', va='bottom', color='#555555', fontweight='bold')

        # Exact
        ax.text(24.74, 7043 + 150, "Exact CG\n(24.7s, $7043)", ha='center', va='bottom', color=COLORS['Exact'],
                fontweight='bold')

        # MIP
        ax.text(15.00, 7550 + 150, "MIP (15s)\n$7550", ha='center', va='bottom', color=COLORS['MIP'], fontweight='bold')

        # GNN (我们的主角，标签放下面，避开 MIP)
        ax.text(15.26, 7043 - 350, "GNN-CG (Ours)\n(15.3s, $7043)", ha='center', va='top', color=COLORS['GNN'],
                fontweight='bold')

        # 6. 绘制核心箭头 (Better Solution in Same Time)
        # 使用弯曲的箭头，看起来更灵活
        arrow = mpatches.FancyArrowPatch(
            (15.00, 7550), (15.26, 7150),  # 起点(MIP) -> 终点(GNN上方一点)
            connectionstyle="arc3,rad=-0.2",  # 弯曲程度
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color='black',
            zorder=20
        )
        ax.add_patch(arrow)

        # 箭头的文字说明 (加个背景框，显得专业)
        ax.text(12.5, 8000,
                "Same Time,\nBetter Quality!",
                ha='center', va='center',
                color='#D62728', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D62728", alpha=0.9))

        # 7. 坐标轴标签与范围
        ax.set_xlabel("Computation Time (seconds)", fontweight='bold')
        ax.set_ylabel("Total Operational Cost ($)", fontweight='bold')
        ax.set_title("Algorithm Performance Comparison (N=100)", fontweight='bold', pad=20)

        # 适当留白
        ax.set_xlim(-2, 30)
        ax.set_ylim(6500, 11000)

        # 8. 美化图例 (去掉边框，放在最佳位置)
        legend = ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', fancybox=True)

        # 保存
        plt.tight_layout()
        plt.savefig("Fig_Pareto_Journal_Style.png", dpi=600)  # 600 DPI 超高清
        print("✅ 顶刊级美图已生成: Fig_Pareto_Journal_Style.png")

    except Exception as e:
        print(f"绘图失败: {e}")


if __name__ == "__main__":
    plot_beautiful_pareto()