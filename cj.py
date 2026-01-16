import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置绘图风格 (Windows 专用修复: 使用 Segoe UI Emoji 支持图标)
plt.rcParams['font.family'] = 'Segoe UI Emoji'
plt.rcParams['font.sans-serif'] = ['Segoe UI Emoji', 'Arial']
plt.rcParams['axes.linewidth'] = 1.5

def draw_trimodal_port_scenario():
    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    
    # --- 1. 环境层 (Environment) ---
    sea = patches.Rectangle((0, 0), 14, 3.5, color='#E0F7FA', alpha=1, zorder=0)
    ax.add_patch(sea)
    land = patches.Rectangle((0, 3.5), 14, 3.5, color='#ECEFF1', alpha=1, zorder=0)
    ax.add_patch(land)
    ax.axhline(y=3.5, color='#37474F', linewidth=4, zorder=1)
    
    # --- 2. 绘制船舶通用函数 ---
    def draw_ship_iconic(x, y, name, color, mode_type):
        # 船体
        ship_body = patches.Polygon([[x, y], [x+3, y], [x+2.8, y-0.8], [x+0.2, y-0.8]], 
                                    closed=True, facecolor='white', edgecolor=color, linewidth=2, zorder=2)
        ax.add_patch(ship_body)
        cabin = patches.Rectangle((x+0.3, y), 2.4, 0.6, facecolor='white', edgecolor=color, linewidth=2, zorder=2)
        ax.add_patch(cabin)
        # 船名
        ax.text(x+1.5, y-0.4, name, ha='center', va='center', fontsize=9, fontweight='bold', color=color)
        
        # 根据模式绘制不同特征
        if mode_type == 'SP': # Shore Power
            ax.text(x+1.5, y+0.3, "⚡", ha='center', va='center', fontsize=14, color=color)
        elif mode_type == 'BS': # Battery Swap
            ax.text(x+1.5, y+0.3, "🔋", ha='center', va='center', fontsize=14, color=color)
        elif mode_type == 'AE': # Brown Mode (Auxiliary Engine)
            # 画烟囱
            chimney = patches.Rectangle((x+2.0, y+0.6), 0.2, 0.4, facecolor='#5D4037', zorder=1)
            ax.add_patch(chimney)
            # 画废气 (Smoke) - 暗示污染
            ax.text(x+2.2, y+1.3, "☁️", ha='center', va='center', fontsize=20, color='#616161', alpha=0.8)
            ax.text(x+2.5, y+1.6, "☁️", ha='center', va='center', fontsize=15, color='#757575', alpha=0.6)

    # --- 3. 绘制三艘代表性船舶 ---

    # Scenario 1: Shore Power (Green, Fixed)
    draw_ship_iconic(1.0, 2.5, "Vessel A\n(Fixed Berth)", '#00796B', 'SP')
    # 设施: 充电桩
    sp_box = patches.Rectangle((2.3, 3.5), 0.4, 0.4, facecolor='#B2DFDB', edgecolor='#00796B', zorder=3)
    ax.add_patch(sp_box)
    ax.text(2.5, 3.7, "Station", ha='center', va='center', fontsize=7, color='#00796B')
    # 连接线
    ax.plot([2.5, 2.5], [3.5, 3.1], color='#00796B', linewidth=3, linestyle='-')
    
    # Scenario 2: Battery Swapping (Green, Mobile)
    draw_ship_iconic(5.5, 2.5, "Vessel B\n(Anywhere)", '#D84315', 'BS')
    # 设施: 换电车
    ax.text(6.5, 3.7, "🚚", ha='center', va='center', fontsize=30, color='#D84315', zorder=5)
    ax.text(7.0, 3.1, "🔋", ha='center', va='center', fontsize=15, color='#2E7D32', zorder=6)

    # Scenario 3: Brown Mode (Brown, Fallback) - NEW!
    draw_ship_iconic(10.0, 2.5, "Vessel C\n(Brown Mode)", '#5D4037', 'AE')
    # 无外部设施，但有排放
    
    # --- 4. 绘制岸桥 (SIMOPS Context) ---
    def add_crane(x):
        ax.text(x, 4.2, "🏗️", ha='center', va='center', fontsize=45, color='#455A64', zorder=3)
        ax.plot([x, x], [4.0, 3.2], color='#455A64', linestyle=':', linewidth=2, zorder=4)

    add_crane(2.5)  # Ship A Ops
    add_crane(7.0)  # Ship B Ops
    add_crane(11.5) # Ship C Ops (Brown mode also has SIMOPS!)

    # --- 5. 标注框 (Annotations) ---
    # SP
    ax.text(2.5, 6.0, "Shore Power:\nFixed & Clean\n(Constraint: Plugs)", 
            ha='center', va='top', fontsize=9, color='#004D40',
            bbox=dict(boxstyle="round,pad=0.3", fc="#E0F2F1", ec="#00796B"))

    # BS
    ax.text(7.0, 6.0, "Battery Swap:\nMobile & Clean\n(Constraint: Trucks)", 
            ha='center', va='top', fontsize=9, color='#BF360C',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FBE9E7", ec="#D84315"))

    # AE (Brown)
    ax.text(11.5, 6.0, "Auxiliary Engine:\nHigh Cost & Emission\n(No Constraint: Zero Wait)", 
            ha='center', va='top', fontsize=9, color='#3E2723',
            bbox=dict(boxstyle="round,pad=0.3", fc="#EFEBE9", ec="#5D4037"))

    # 泊位线
    for x in [4.5, 9.0, 13.5]:
        ax.plot([x, x], [3.5, 2.0], color='white', linestyle='--', linewidth=2, zorder=1)

    # 隐藏坐标轴
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('Fig1_Port_Scenario_TriModal.png', bbox_inches='tight')
    return "Tri-modal plot generated!"

# 运行绘图
draw_trimodal_port_scenario()