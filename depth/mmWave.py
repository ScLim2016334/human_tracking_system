import serial
import time
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re

# ---------- 串口配置 ----------
COM_PORT = 'COM12'           # 你的串口号
BAUD_RATE = 115200

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"[INFO] 串口 {COM_PORT} 已打开")
    time.sleep(1)
except Exception as e:
    print(f"[ERROR] 无法打开串口：{e}")
    exit()

# ---------- 画图初始化 ----------
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter([], [], s=100, c='red')
ax.set_xlim(-2, 2)
ax.set_ylim(0, 4)
ax.set_title("实时 2D 雷达图")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.grid(True)

# ---------- 极坐标转笛卡尔 ----------
def polar_to_xy(r, theta_deg=90):
    theta = math.radians(theta_deg)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

# ---------- 更新函数 ----------
buffer = ""
def update(frame):
    global buffer
    # 尝试读取串口数据
    if ser.in_waiting > 0:
        raw = ser.read(ser.in_waiting).decode(errors='ignore')
        buffer += raw

        # 使用正则匹配 "Range XX"
        matches = re.findall(r"Range\s+(\d+)", buffer)
        if matches:
            latest = int(matches[-1])  # 取最近的一个
            distance_m = latest / 100.0  # 转换为米

            x, y = polar_to_xy(distance_m)
            sc.set_offsets([[x, y]])
            print(f"[INFO] 目标距离: {latest} cm")

        # 清理 buffer 防止越积越多
        if len(buffer) > 100:
            buffer = buffer[-50:]

ani = FuncAnimation(fig, update, interval=100)
plt.show()
