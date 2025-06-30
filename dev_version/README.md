# Human Tracking System - IoU-based Occlusion Detection

这是一个重构后的人体跟踪系统，具有人脸识别和基于IoU的遮挡处理功能。新的遮挡策略更加稳定和可靠。

## 新的遮挡处理策略

### 1. IoU检测机制
- **检测原理**: 通过计算主人骨架与其他骨架的IoU（交并比）来检测遮挡
- **阈值设置**: `OCCLUSION_IOU_THRESHOLD = 0.3` - 当IoU超过此值时认为发生遮挡
- **稳定性**: 需要连续5帧IoU都超过阈值才确认遮挡，避免误判

### 2. 语音提示系统
系统会根据不同情况发出语音提示：

#### 遮挡检测时
- **提示**: "Master please stop for X seconds" (主人请停止X秒钟)
- **目的**: 让主人和遮挡者都停止移动，等待遮挡结束

#### 恢复阶段
- **单骨架情况**: 自动识别为主人，开始跟随
- **多骨架情况**: "Master please look at me" (主人请看着我)
- **无骨架情况**: "Master I can't see you, please come in front of me" (主人我看不到您，请到我前面来)

### 3. 恢复机制
1. **等待时间**: 遮挡结束后等待3秒让场景稳定
2. **场景分析**: 分析视野内的骨架数量
3. **智能恢复**: 根据骨架数量选择不同的恢复策略

## 模块结构

### 1. `config.py` - 配置文件
新增的IoU遮挡检测参数：
- `OCCLUSION_IOU_THRESHOLD`: IoU检测阈值
- `OCCLUSION_STOP_TIME`: 停止等待时间
- `OCCLUSION_DETECTION_FRAMES`: 确认遮挡的连续帧数
- `DISTANCE_THRESHOLD_FOR_TRACKING`: 跟踪距离阈值

### 2. `state_manager.py` - 状态管理器
新增的遮挡状态管理：
- `occlusion_detected`: 当前是否检测到遮挡
- `occlusion_iou_frames`: IoU超过阈值的连续帧数
- `occlusion_recovery_mode`: 是否处于恢复模式
- `occlusion_stop_announced`: 是否已发出停止提示

### 3. `voice_announcer.py` - 语音提示模块
处理各种语音公告：
- 支持中英文双语
- 多种TTS引擎支持（pyttsx3, Windows SAPI, espeak）
- 防重复播放机制

### 4. `skeleton_tracker.py` - 骨架跟踪器
核心的IoU遮挡检测逻辑：
- `detect_iou_occlusion()`: IoU遮挡检测
- `handle_occlusion_recovery()`: 遮挡恢复处理
- `get_skeletons_in_tracking_range()`: 获取跟踪范围内的骨架

### 5. `visualization.py` - 可视化模块
增强的显示功能：
- 显示IoU数值
- 遮挡状态信息
- 恢复模式计时器

## 使用方法

### 运行程序
```bash
python human_tracking_refactored.py
```

### 操作说明

#### 步骤1：记录主人人脸
1. 靠近摄像头
2. 按 's' 键多次记录不同角度的人脸
3. 按 'n' 键进入下一步

#### 步骤2：校准主人骨架
1. 远离摄像头，确保全身可见
2. 做出"双手叉腰"动作
3. 系统会自动识别并校准

#### 步骤3：正常跟踪模式
- 系统会自动跟踪主人（红色边界框）
- 其他骨架显示为绿色边界框
- 当检测到遮挡时，会显示IoU数值
- 系统会自动处理遮挡恢复

### 按键控制
- `s`: 记录人脸（步骤1）
- `n`: 进入下一步
- `r`: 重置所有记录
- `q`: 退出程序

## 遮挡处理流程

### 1. 遮挡检测
```
主人骨架 + 其他骨架 → 计算IoU → IoU > 0.3 → 连续5帧确认 → 遮挡检测成功
```

### 2. 语音提示
```
遮挡检测成功 → 播放"停止"提示 → 等待3秒 → 进入恢复模式
```

### 3. 恢复策略
```
恢复模式 → 分析骨架数量:
├── 0个骨架 → "请到前面来" → 等待主人出现
├── 1个骨架 → 自动识别为主人 → 开始跟随
└── 多个骨架 → "请看着我" → 人脸识别 → 绑定骨架
```

## 配置参数

### IoU遮挡检测
- `OCCLUSION_IOU_THRESHOLD = 0.3`: IoU检测阈值
- `OCCLUSION_STOP_TIME = 3`: 停止等待时间（秒）
- `OCCLUSION_DETECTION_FRAMES = 5`: 确认遮挡的连续帧数

### 跟踪距离
- `DISTANCE_THRESHOLD_FOR_TRACKING = 200`: 跟踪距离阈值（像素）

### 语音设置
- `ENABLE_VOICE_ANNOUNCEMENTS = True`: 启用语音提示
- `VOICE_LANGUAGE = "en"`: 语音语言（"en"或"zh"）

## 优势

1. **更稳定的遮挡检测**: 基于IoU而非距离，不受深度估计影响
2. **智能恢复机制**: 根据场景自动选择最佳恢复策略
3. **语音交互**: 通过语音提示与主人交互
4. **可视化反馈**: 实时显示IoU数值和遮挡状态
5. **防误判机制**: 需要连续多帧确认才触发遮挡处理

## 依赖库

- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- face_recognition (`face_recognition`)
- pyttsx3 (`pyttsx3`) - 可选，用于语音合成

## 安装语音依赖（可选）

```bash
# 安装pyttsx3（跨平台TTS）
pip install pyttsx3

# 或安装Windows SAPI支持
pip install pywin32

# 或安装espeak（Linux）
sudo apt-get install espeak
``` 