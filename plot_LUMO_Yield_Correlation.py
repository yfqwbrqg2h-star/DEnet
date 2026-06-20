import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 完全同步最新截图所有LUMO原始数值，分组不变
data = {
    "label": ["3a", "3b", "3c", "3d", "3e", "3f", "3g", "3h", "3i", "3j", "3k", "3l", "3m", "3n", "3o"],
    "lumo": [
        -0.794558,  # 3a
        -0.877745,  # 3b
        -0.316933,  # 3c
        -0.601411,  # 3d
        -0.671510,  # 3e 更新
        -0.256751,  # 3f
        -0.411041,  # 3g 更新
        -1.271834,  # 3h
        -1.071581,  # 3i
        -0.858636,  # 3j
        -0.967500,  # 3k
        -1.178155,  # 3l
        -1.258085,  # 3m
        -0.871960,  # 3n 更新（吡啶，参与回归）
        -0.252638   # 3o 更新（噻吩，不参与回归）
    ],
    "yield": [86, 82, 80, 81, 77, 78, 75, 88, 85, 84, 90, 93, 94, 72, 63],
    "group": [
        "Neutral parent",
        "Electron-donating",
        "Electron-donating",
        "Electron-donating",
        "Electron-donating",
        "Electron-donating",
        "Electron-donating",
        "Weak electron-withdrawing (halogen)",
        "Weak electron-withdrawing (halogen)",
        "Weak electron-withdrawing (halogen)",
        "Strong electron-withdrawing",
        "Strong electron-withdrawing",
        "Strong electron-withdrawing",
        "Heterocycle (electron-deficient)",
        "Heterocycle (electron-rich)"
    ]
}
df = pd.DataFrame(data)

# 回归规则：包含3a~3n（吡啶参与），仅剔除3o噻吩
df_reg = df[df["label"] != "3o"]
slope, intercept, r, p, se = linregress(df_reg["lumo"], df_reg["yield"])
r2 = r ** 2

# 期刊配色
color_map = {
    "Neutral parent": "#2563eb",
    "Electron-donating": "#f97316",
    "Weak electron-withdrawing (halogen)": "#16a34a",
    "Strong electron-withdrawing": "#dc2626",
    "Heterocycle (electron-deficient)": "#6b7280",
    "Heterocycle (electron-rich)": "#8b5cf6"
}

# 绘图全局设置
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 6), dpi=120)

# 绘制全部15个底物散点（3o噻吩依然可视化展示）
drawn_groups = set()
for _, row in df.iterrows():
    g = row["group"]
    c = color_map[g]
    show_label = g if g not in drawn_groups else ""
    plt.scatter(
        row["lumo"], row["yield"],
        color=c, s=120, edgecolors="black", linewidth=0.8,
        label=show_label, zorder=5
    )
    drawn_groups.add(g)
    # 底物编号文字标注
    plt.text(row["lumo"] + 0.022, row["yield"] + 0.35, row["label"], fontsize=8.5)

# 拟合线范围基于回归样本（不含3o）
x_fit_range = [df_reg["lumo"].min() - 0.15, df_reg["lumo"].max() + 0.15]
y_fit_range = [slope * x + intercept for x in x_fit_range]
plt.plot(
    x_fit_range, y_fit_range,
    color="#111111", linestyle="--", linewidth=1.8,
    label=f"Linear fit (exclude thiophene 3o), $R^2$ = {r2:.3f}"
)

# 坐标轴与标题
plt.xlabel("Predicted LUMO Energy / eV", fontsize=12)
plt.ylabel("Isolated Yield / %", fontsize=12)
plt.title("Correlation between Predicted LUMO Energy and Catalytic Yield", fontsize=13, pad=12)
plt.legend(loc="lower right", fontsize=8.5)
plt.grid(alpha=0.25, linestyle="--")
plt.tight_layout()

# 保存高清图
plt.savefig("lumo_yield_updated_lumo_data.png", dpi=300, bbox_inches="tight")
plt.show()

# 控制台输出回归统计
print("="*50)
print("Linear Regression Result (Include pyridine 3n, exclude thiophene 3o)")
print("="*50)
print(f"Slope (斜率)        = {slope:.4f}")
print(f"Intercept (截距)    = {intercept:.4f}")
print(f"Correlation r       = {r:.4f}")
print(f"R-squared (R²)      = {r2:.4f}")
print(f"P-value (显著性)    = {p:.4e}")
print("="*50)
