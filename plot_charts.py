import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置论文风格的绘图参数
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Times New Roman'  # 论文常用字体
plt.rcParams['font.size'] = 12


def plot_ablation():
    df = pd.read_csv("exp_ablation_results.csv")

    plt.figure(figsize=(8, 6))

    # 绘制柱状对比图
    sns.barplot(data=df, x="Scale", y="Time", hue="Method", palette="viridis")

    plt.xlabel("Number of Ships (Problem Scale)", fontweight='bold')
    plt.ylabel("Computation Time (Seconds)", fontweight='bold')
    plt.title("Computational Efficiency: AI-Accelerated vs. Exact CG", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Algorithm Strategy")

    plt.savefig("Fig_Ablation_Time.png", dpi=300, bbox_inches='tight')
    print("🖼️ 图表 1 生成完毕: Fig_Ablation_Time.png")


def plot_sensitivity():
    df = pd.read_csv("exp_sensitivity_results.csv")

    plt.figure(figsize=(8, 6))

    # 绘制折线图
    plt.plot(df["Battery_Cost"], df["Shore_Rate"], marker='o', linewidth=2, color='#d62728')

    # 画一条岸电成本的参考线 (Cost=50)
    plt.axvline(x=50, color='gray', linestyle='--', label='Shore Power Cost (Ref)')

    plt.xlabel("Battery Swapping Cost ($)", fontweight='bold')
    plt.ylabel("Shore Power Utilization Rate (%)", fontweight='bold')
    plt.title("Sensitivity Analysis: Impact of Energy Price Gap", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig("Fig_Sensitivity_Price.png", dpi=300, bbox_inches='tight')
    print("🖼️ 图表 2 生成完毕: Fig_Sensitivity_Price.png")


if __name__ == "__main__":
    try:
        plot_ablation()
        plot_sensitivity()
        print("\n🎉 所有图表已生成！请查看项目文件夹。")
    except Exception as e:
        print(f"绘图出错，请检查是否有 CSV 文件: {e}")