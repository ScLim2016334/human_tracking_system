# -*- coding: utf-8 -*-
"""
UWB上位机程序，支持标签定位信息命令和心跳包信息命令
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import serial.tools.list_ports
from datetime import datetime
import threading
import queue
import time
import binascii
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import math
from collections import deque

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0, max_change=None):
        self.process_variance = process_variance        # 过程噪声方差
        self.measurement_variance = measurement_variance  # 测量噪声方差
        self.estimate = initial_value                  # 当前估计值
        self.estimate_error = 1                        # 估计误差
        self.last_estimate = initial_value             # 上一次的估计值
        self.max_change = max_change                   # 最大允许变化量
        self.initialized = False                       # 是否已初始化
        
    def update(self, measurement):
        # 首次测量时直接初始化
        if not self.initialized:
            self.estimate = measurement
            self.last_estimate = measurement
            self.initialized = True
            return measurement
            
        # 对测量值进行预处理，限制突变
        if self.max_change is not None:
            max_diff = self.max_change
            measurement = max(min(measurement, 
                                self.last_estimate + max_diff),
                                self.last_estimate - max_diff)
        
        # 预测步骤
        prediction = self.last_estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # 更新步骤
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        # 保存本次估计值
        self.last_estimate = self.estimate
        
        return self.estimate
        
    def reset(self):
        """重置滤波器状态"""
        self.estimate = 0
        self.estimate_error = 1
        self.last_estimate = 0
        self.initialized = False

class PlotFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # 创建图形
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111, projection='polar')
        
        # 设置角度范围为前方180度
        self.ax.set_theta_zero_location('N')  # 设置0度在正北方向
        self.ax.set_theta_direction(-1)  # 设置角度顺时针增加
        self.ax.set_thetamin(-90)  # 设置最小角度为-90度
        self.ax.set_thetamax(90)   # 设置最大角度为90度
        
        # 设置距离范围和网格
        self.ax.set_rlim(0, 500)  # 设置距离范围（厘米）
        self.ax.set_rticks([100, 200, 300, 400, 500])  # 设置径向网格线
        self.ax.set_rlabel_position(0)  # 设置径向标签位置
        
        # 设置角度网格
        self.ax.set_thetagrids([-90, -45, 0, 45, 90])  # 设置角度网格线
        
        # 调整图形布局
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # 创建散点图
        self.scatter = self.ax.scatter([], [], c='red', s=100)
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
    def update_plot(self, distance, azimuth):
        """更新图形显示"""
        # 角度转换：将-180到180度的范围转换为-90到90度
        if azimuth > 180:
            azimuth = azimuth - 360
            
        # 检查角度是否在显示范围内
        if azimuth < -90 or azimuth > 90:
            return
            
        # 更新散点位置
        azimuth_rad = math.radians(azimuth)
        self.scatter.set_offsets(np.c_[azimuth_rad, distance])
        
        # 清除之前的点并重绘
        self.ax.draw_artist(self.scatter)
        self.canvas.draw()
        
    def clear_plot(self):
        """清除图形显示"""
        self.scatter.set_offsets(np.c_[[], []])
        self.canvas.draw()

class SerialLogger:
    def __init__(self, port=None, baudrate=230400, output_file="data.txt"):
        self.port = port
        self.baudrate = baudrate
        self.output_file = output_file
        self.serial = None
        self.data_buffer = bytearray()
        self.HEADER = b'\xff\xff\xff\xff'
        
        # 添加命令码常量
        self.CMD_LOCATION = 0x2001
        self.CMD_HEARTBEAT = 0x2002
        
        # 更新两种数据包的长度常量
        self.LOCATION_PACKET_LENGTH = 37  # 定位包长度
        self.HEARTBEAT_PACKET_LENGTH = 16  # 心跳包长度
        
        # 添加缓存大小限制
        self.MAX_BUFFER_SIZE = 1024 * 10  # 10KB
        
    def list_available_ports(self):
        """列出所有可用的串口"""
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("没有找到可用的串口!")
            return []
        
        print("\n可用的串口:")
        for port in ports:
            print(f"- {port.device}: {port.description}")
        return ports

    def initialize_serial(self):
        """初始化串口连接"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            print(f"\n成功打开串口 {self.port}")
            return True
        except serial.SerialException as e:
            print(f"打开串口失败: {str(e)}")
            return False

    def process_buffer(self):
        """优化的数据缓冲区处理，支持两种类型数据包"""
        # 限制缓冲区大小
        if len(self.data_buffer) > self.MAX_BUFFER_SIZE:
            self.data_buffer = self.data_buffer[-self.LOCATION_PACKET_LENGTH:]
            
        while True:
            header_pos = self.data_buffer.find(self.HEADER)
            if header_pos == -1 or len(self.data_buffer) < 10:  # 至少需要头部+长度+命令码
                break

            if header_pos > 0:
                self.data_buffer = self.data_buffer[header_pos:]
                
            # 确保有足够的数据读取命令码
            if len(self.data_buffer) >= 10:
                # 读取命令码
                command_code = int.from_bytes(self.data_buffer[8:10], 'big')
                
                # 根据命令码确定包长度
                if command_code == self.CMD_LOCATION and len(self.data_buffer) >= self.LOCATION_PACKET_LENGTH:
                    packet = self.data_buffer[:self.LOCATION_PACKET_LENGTH]
                    self.data_buffer = self.data_buffer[self.LOCATION_PACKET_LENGTH:]
                    return packet, "location"
                    
                elif command_code == self.CMD_HEARTBEAT and len(self.data_buffer) >= self.HEARTBEAT_PACKET_LENGTH:
                    packet = self.data_buffer[:self.HEARTBEAT_PACKET_LENGTH]
                    self.data_buffer = self.data_buffer[self.HEARTBEAT_PACKET_LENGTH:]
                    return packet, "heartbeat"
                    
                else:
                    # 如果不是已知命令码或数据不足，丢弃这个头部
                    self.data_buffer = self.data_buffer[4:]
            else:
                # 数据不足，等待更多数据
                break

        return None, None

    def parse_location_packet(self, packet):
        """解析定位数据包，返回解析后的字段"""
        if len(packet) != self.LOCATION_PACKET_LENGTH:
            return None

        try:
            fields = {
                'MessageHeader': packet[0:4],
                'PacketLength': int.from_bytes(packet[4:6], 'big'),
                'SequenceID': int.from_bytes(packet[6:8], 'big'),
                'RequestCommand': int.from_bytes(packet[8:10], 'big'),
                'VersionID': int.from_bytes(packet[10:12], 'big'),
                'AnchorID': int.from_bytes(packet[12:16], 'big'),
                'TagID': int.from_bytes(packet[16:20], 'big'),
                'Distance': int.from_bytes(packet[20:24], 'big'),
                'Azimuth': int.from_bytes(packet[24:26], 'big', signed=True),
                'Elevation': int.from_bytes(packet[26:28], 'big', signed=True),
                'TagStatus': int.from_bytes(packet[28:30], 'big'),
                'BatchSn': int.from_bytes(packet[30:32], 'big'),
                'Reserve': packet[32:36],
                'XorByte': packet[36],
                'PacketType': 'location'  # 添加包类型标记
            }
            return fields
        except Exception as e:
            print(f"解析定位数据包错误: {str(e)}")
            return None
            
    def parse_heartbeat_packet(self, packet):
        """解析心跳包数据包，返回解析后的字段"""
        if len(packet) != self.HEARTBEAT_PACKET_LENGTH:
            return None

        try:
            fields = {
                'MessageHeader': packet[0:4],
                'PacketLength': int.from_bytes(packet[4:6], 'big'),
                'SequenceID': int.from_bytes(packet[6:8], 'big'),
                'RequestCommand': int.from_bytes(packet[8:10], 'big'),
                'VersionID': int.from_bytes(packet[10:12], 'big'),
                'AnchorID': int.from_bytes(packet[12:16], 'big'),
                'PacketType': 'heartbeat'  # 添加包类型标记
            }
            return fields
        except Exception as e:
            print(f"解析心跳包错误: {str(e)}")
            return None

    def close(self):
        """关闭串口连接"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print(f"串口 {self.port} 已关闭")

class SerialGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UWB串口数据监控")
        self.root.geometry("1200x800")
        
        # 使用双缓冲队列
        self.data_queue = queue.Queue(maxsize=100)  # 限制队列大小
        self.log_queue = queue.Queue(maxsize=50)    # 限制日志队列大小
        
        # 添加状态变量，记录当前数据包类型
        self.current_packet_type = None
        self.last_heartbeat_time = 0
        
        # 卡尔曼滤波器参数变量
        self.distance_process_var = tk.DoubleVar(value=0.01)
        self.distance_measure_var = tk.DoubleVar(value=50)
        self.distance_max_change = tk.DoubleVar(value=100)
        
        self.angle_process_var = tk.DoubleVar(value=0.01)
        self.angle_measure_var = tk.DoubleVar(value=30)
        self.angle_max_change = tk.DoubleVar(value=30)
        
        # 添加滤波器开关状态
        self.filter_enabled = tk.BooleanVar(value=True)
        
        # 初始化滤波器
        self.init_filters()
        
        # 创建界面
        self.create_frames()
        self.create_widgets()
        
        # 串口管理器实例
        self.serial_logger = SerialLogger()
        self.is_logging = False
        
        # 更新显示的计时器
        self.last_gui_update = time.time()
        self.last_plot_update = time.time()
        self.GUI_UPDATE_INTERVAL = 0.05  # 20Hz
        self.PLOT_UPDATE_INTERVAL = 0.1  # 10Hz
        
        # 刷新可用串口
        self.update_ports()
        
        # 启动更新循环
        self.update_display()

    def init_filters(self):
        """初始化卡尔曼滤波器"""
        self.distance_filter = KalmanFilter(
            process_variance=self.distance_process_var.get(),
            measurement_variance=self.distance_measure_var.get(),
            max_change=self.distance_max_change.get()
        )
        
        self.azimuth_filter = KalmanFilter(
            process_variance=self.angle_process_var.get(),
            measurement_variance=self.angle_measure_var.get(),
            max_change=self.angle_max_change.get()
        )
        
        self.elevation_filter = KalmanFilter(
            process_variance=self.angle_process_var.get(),
            measurement_variance=self.angle_measure_var.get(),
            max_change=self.angle_max_change.get()
        )

    def create_frames(self):
        """创建主要框架"""
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_widgets(self):
        """创建控件"""
        # 左侧控制和数据显示区域
        # 顶部控制区
        control_frame = ttk.LabelFrame(self.left_frame, text="控制面板", padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 串口控制区域
        serial_frame = ttk.Frame(control_frame)
        serial_frame.pack(fill=tk.X, pady=5)
        
        # 串口选择
        ttk.Label(serial_frame, text="串口:").grid(row=0, column=0, padx=5)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var)
        self.port_combo.grid(row=0, column=1, padx=5)
        
        # 刷新按钮
        ttk.Button(serial_frame, text="刷新串口", command=self.update_ports).grid(row=0, column=2, padx=5)
        
        # 波特率选择
        ttk.Label(serial_frame, text="波特率:").grid(row=0, column=3, padx=5)
        self.baud_var = tk.StringVar(value="230400")
        baud_combo = ttk.Combobox(serial_frame, textvariable=self.baud_var, 
                                 values=["9600", "115200", "230400", "460800"])
        baud_combo.grid(row=0, column=4, padx=5)
        
        # 开始/停止按钮
        self.start_button = ttk.Button(serial_frame, text="开始", command=self.toggle_logging)
        self.start_button.grid(row=0, column=5, padx=5)
        
        # 清除按钮
        ttk.Button(serial_frame, text="清除显示", command=self.clear_display).grid(row=0, column=6, padx=5)
        
        # 滤波器参数设置区域
        filter_frame = ttk.LabelFrame(control_frame, text="滤波器参数设置", padding="5")
        filter_frame.pack(fill=tk.X, pady=5)
        
        # 添加滤波器开关
        filter_toggle_frame = ttk.Frame(filter_frame)
        filter_toggle_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(filter_toggle_frame, text="滤波器状态:").pack(side=tk.LEFT, padx=5)
        self.filter_toggle = ttk.Checkbutton(
            filter_toggle_frame, 
            text="启用滤波", 
            variable=self.filter_enabled,
            command=self.toggle_filter
        )
        self.filter_toggle.pack(side=tk.LEFT, padx=5)
        
        # 距离滤波器参数
        distance_frame = ttk.LabelFrame(filter_frame, text="距离滤波器", padding="5")
        distance_frame.pack(fill=tk.X, pady=2)
        
        # 创建参数调整行
        self.create_parameter_row(distance_frame, "过程噪声:", self.distance_process_var, 0, 0.001, 1.0, 0)
        self.create_parameter_row(distance_frame, "测量噪声:", self.distance_measure_var, 0, 1.0, 200.0, 1)
        self.create_parameter_row(distance_frame, "最大变化:", self.distance_max_change, 0, 1.0, 500.0, 2)
        
        # 角度滤波器参数
        angle_frame = ttk.LabelFrame(filter_frame, text="角度滤波器", padding="5")
        angle_frame.pack(fill=tk.X, pady=2)
        
        self.create_parameter_row(angle_frame, "过程噪声:", self.angle_process_var, 0, 0.001, 1.0, 0)
        self.create_parameter_row(angle_frame, "测量噪声:", self.angle_measure_var, 0, 1.0, 200.0, 1)
        self.create_parameter_row(angle_frame, "最大变化:", self.angle_max_change, 0, 1.0, 180.0, 2)
        
        # 应用按钮
        ttk.Button(filter_frame, text="应用参数", command=self.apply_filter_params).pack(pady=5)
        
        # 数据显示区
        display_frame = ttk.LabelFrame(self.left_frame, text="数据显示", padding="5")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 基站信息显示
        station_frame = ttk.LabelFrame(display_frame, text="基站信息", padding="5")
        station_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 数据包类型显示
        ttk.Label(station_frame, text="数据包类型:").grid(row=0, column=0, padx=5)
        self.packet_type_var = tk.StringVar(value="--")
        ttk.Label(station_frame, textvariable=self.packet_type_var).grid(row=0, column=1, padx=5)
        
        # 基站ID显示
        ttk.Label(station_frame, text="基站ID:").grid(row=0, column=2, padx=5)
        self.anchor_id_var = tk.StringVar(value="--")
        ttk.Label(station_frame, textvariable=self.anchor_id_var).grid(row=0, column=3, padx=5)
        
        # 标签ID显示
        ttk.Label(station_frame, text="标签ID:").grid(row=0, column=4, padx=5)
        self.tag_id_var = tk.StringVar(value="--")
        ttk.Label(station_frame, textvariable=self.tag_id_var).grid(row=0, column=5, padx=5)
        
        # 实时数据显示
        data_frame = ttk.LabelFrame(display_frame, text="实时数据", padding="5")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 距离显示
        ttk.Label(data_frame, text="距离:").grid(row=0, column=0, padx=5)
        self.distance_var = tk.StringVar(value="-- cm")
        ttk.Label(data_frame, textvariable=self.distance_var).grid(row=0, column=1, padx=5)
        
        # 方位角显示
        ttk.Label(data_frame, text="方位角:").grid(row=0, column=2, padx=5)
        self.azimuth_var = tk.StringVar(value="-- °")
        ttk.Label(data_frame, textvariable=self.azimuth_var).grid(row=0, column=3, padx=5)
        
        # 仰角显示
        ttk.Label(data_frame, text="仰角:").grid(row=0, column=4, padx=5)
        self.elevation_var = tk.StringVar(value="-- °")
        ttk.Label(data_frame, textvariable=self.elevation_var).grid(row=0, column=5, padx=5)
        
        # 日志显示区
        self.log_text = scrolledtext.ScrolledText(display_frame, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧添加可视化区域
        plot_frame = ttk.LabelFrame(self.right_frame, text="位置可视化", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建极坐标图
        self.plot = PlotFrame(plot_frame)
        self.plot.pack(fill=tk.BOTH, expand=True)

    def create_parameter_row(self, parent, label, variable, column, min_val, max_val, row):
        """创建参数调整行"""
        ttk.Label(parent, text=label).grid(row=row, column=column, padx=5)
        spinbox = ttk.Spinbox(
            parent,
            from_=min_val,
            to=max_val,
            increment=0.001,
            textvariable=variable,
            width=10
        )
        spinbox.grid(row=row, column=column+1, padx=5)
        
        # 添加滑动条
        scale = ttk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            variable=variable,
            orient=tk.HORIZONTAL
        )
        scale.grid(row=row, column=column+2, padx=5, sticky='ew')
        
        parent.grid_columnconfigure(column+2, weight=1)

    def toggle_filter(self):
        """切换滤波器状态"""
        if self.filter_enabled.get():
            self.log_message("已启用滤波器")
            # 重置滤波器状态
            self.init_filters()
        else:
            self.log_message("已禁用滤波器")

    def apply_filter_params(self):
        """应用新的滤波器参数"""
        # 重新初始化滤波器
        self.init_filters()
        self.log_message("已更新滤波器参数")

    def update_ports(self):
        """更新可用串口列表"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])

    def toggle_logging(self):
        """切换数据记录状态"""
        if not self.is_logging:
            self.distance_filter.reset()
            self.azimuth_filter.reset()
            self.elevation_filter.reset()
            # 开始记录
            self.serial_logger.port = self.port_var.get()
            self.serial_logger.baudrate = int(self.baud_var.get())
            
            if self.serial_logger.initialize_serial():
                self.is_logging = True
                self.start_button['text'] = "停止"
                
                # 启动数据记录线程
                self.log_thread = threading.Thread(target=self.logging_thread)
                self.log_thread.daemon = True
                self.log_thread.start()
                
                self.log_message("开始记录数据...")
            else:
                self.log_message("打开串口失败!")
        else:
            # 停止记录
            self.is_logging = False
            self.start_button['text'] = "开始"
            self.serial_logger.close()
            self.log_message("停止记录数据")

    def logging_thread(self):
        """优化的数据记录线程，支持两种数据包类型"""
        while self.is_logging:
            try:
                if self.serial_logger.serial.in_waiting:
                    # 批量读取数据
                    new_data = self.serial_logger.serial.read(
                        min(self.serial_logger.serial.in_waiting, 1024)
                    )
                    self.serial_logger.data_buffer.extend(new_data)
                    
                    # 处理所有可用的数据包
                    while True:
                        packet, packet_type = self.serial_logger.process_buffer()
                        if not packet:
                            break
                            
                        # 根据包类型解析数据
                        fields = None
                        if packet_type == "location":
                            fields = self.serial_logger.parse_location_packet(packet)
                        elif packet_type == "heartbeat":
                            fields = self.serial_logger.parse_heartbeat_packet(packet)
                            self.last_heartbeat_time = time.time()
                            
                        if fields:
                            # 使用非阻塞方式添加到队列
                            try:
                                self.data_queue.put_nowait(fields)
                            except queue.Full:
                                # 队列满时，丢弃最老的数据
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(fields)
                                except queue.Empty:
                                    pass
                                    
                            # 构建日志信息
                            hex_data = binascii.hexlify(packet).decode('utf-8')
                            hex_formatted = ' '.join(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            
                            if packet_type == "location":
                                parsed_info = (
                                    f"基站ID: {fields['AnchorID']}, "
                                    f"标签ID: {fields['TagID']}, "
                                    f"距离: {fields['Distance']}cm, "
                                    f"方位角: {fields['Azimuth']}°, "
                                    f"仰角: {fields['Elevation']}°"
                                )
                            else:  # heartbeat
                                parsed_info = (
                                    f"心跳包 - 基站ID: {fields['AnchorID']}"
                                )
                                
                            log_line = f"[{timestamp}] {hex_formatted}\n解析结果: {parsed_info}\n"
                            
                            # 非阻塞方式添加日志
                            try:
                                self.log_queue.put_nowait(log_line)
                            except queue.Full:
                                pass
                                
                time.sleep(0.001)  # 短暂休眠以降低CPU使用率
                
            except serial.SerialException as e:
                self.log_message(f"串口错误: {str(e)}")
                self.is_logging = False
                break

    def update_display(self):
        """优化的显示更新，支持不同类型的数据包"""
        current_time = time.time()
        
        # 更新GUI显示
        if current_time - self.last_gui_update >= self.GUI_UPDATE_INTERVAL:
            try:
                # 批量处理数据队列
                data_processed = False
                last_fields = None
                
                while not self.data_queue.empty():
                    fields = self.data_queue.get_nowait()
                    last_fields = fields
                    data_processed = True
                    
                    # 更新当前包类型
                    self.current_packet_type = fields['PacketType']
                    self.packet_type_var.set("定位包" if fields['PacketType'] == 'location' else "心跳包")
                    
                    # 更新基站ID
                    self.anchor_id_var.set(f"{fields['AnchorID']}")
                    
                    if fields['PacketType'] == 'location':
                        # 处理定位包
                        if self.filter_enabled.get():
                            # 使用滤波器处理数据
                            filtered_distance = self.distance_filter.update(fields['Distance'])
                            filtered_azimuth = self.azimuth_filter.update(fields['Azimuth'])
                            filtered_elevation = self.elevation_filter.update(fields['Elevation'])
                        else:
                            # 直接使用原始数据
                            filtered_distance = fields['Distance']
                            filtered_azimuth = fields['Azimuth']
                            filtered_elevation = fields['Elevation']
                        
                        # 更新标签ID和位置信息
                        self.tag_id_var.set(f"{fields['TagID']}")
                        self.distance_var.set(f"{int(filtered_distance)} cm")
                        self.azimuth_var.set(f"{int(filtered_azimuth)} °")
                        self.elevation_var.set(f"{int(filtered_elevation)} °")
                        
                        # 更新绘图数据
                        fields['Distance'] = int(filtered_distance)
                        fields['Azimuth'] = int(filtered_azimuth)
                        fields['Elevation'] = int(filtered_elevation)
                        
                    elif fields['PacketType'] == 'heartbeat':
                        # 处理心跳包 - 显示NA
                        self.tag_id_var.set("NA")
                        self.distance_var.set("NA")
                        self.azimuth_var.set("NA")
                        self.elevation_var.set("NA")
                        
                        # 重置滤波器，避免下次收到定位包时使用过时数据
                        if self.filter_enabled.get():
                            self.distance_filter.reset()
                            self.azimuth_filter.reset()
                            self.elevation_filter.reset()
                
                # 只在处理了新数据且达到更新间隔时更新图形
                if data_processed and last_fields and \
                   current_time - self.last_plot_update >= self.PLOT_UPDATE_INTERVAL:
                    # 只在接收到定位包时更新位置图
                    if last_fields['PacketType'] == 'location':
                        self.plot.update_plot(last_fields['Distance'], last_fields['Azimuth'])
                    elif last_fields['PacketType'] == 'heartbeat':
                        # 心跳包时清除图形
                        self.plot.clear_plot()
                    self.last_plot_update = current_time
                    
                # 批量更新日志
                log_text = ""
                while not self.log_queue.empty():
                    try:
                        log_text += self.log_queue.get_nowait()
                    except queue.Empty:
                        break
                        
                if log_text:
                    self.log_text.insert(tk.END, log_text)
                    # 保持显示最新的内容
                    self.log_text.see(tk.END)
                    
                self.last_gui_update = current_time
                
            except queue.Empty:
                pass
                
        # 继续更新循环
        self.root.after(1, self.update_display)

    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}\n"
        try:
            self.log_queue.put_nowait(log_line)
        except queue.Full:
            pass
        
    def clear_display(self):
        """清除显示"""
        self.log_text.delete(1.0, tk.END)
        
    def on_closing(self):
        """关闭窗口时的处理"""
        if self.is_logging:
            self.is_logging = False
            self.serial_logger.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SerialGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()