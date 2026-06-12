# HiFiUTAU-Engine OpenUTAU

基于 PC-NSF-HiFiGAN 神经声码器的高质量歌声合成后端，搭配 [OpenUtau-CustomRenderer分支](https://github.com/xiaobaijunya/OpenUtau-CustomRenderer) 提供的 **CUSTOM_SERVER** 合成引擎使用。

## 简介

HiFiUTAU-Engine 是一个独立运行的 HTTP 语音合成服务，接收 OpenUTAU 发送的 JSON 合成请求，实时合成并返回 WAV 音频。支持 ONNX Runtime 和 PyTorch 两种推理后端，并提供了丰富的参数控制能力，可实现气声、张力、咆哮等多种表现力调节。


## 模型下载

> 请先下载对应模型文件，放置到程序根目录。

### ONNX 模式所需模型

[下载 ONNX 模型包](https://github.com/xiaobaijunya/HiFiUTAU-engine/releases/download/0.0.0/hifiserver_onnx.zip)（含 HiFiGAN 和 HN-SEP 模型）

### PyTorch 模式所需模型

[下载 PyTorch 模型包](https://github.com/xiaobaijunya/HiFiUTAU-engine/releases/download/0.0.1/pytorch_model.zip)（含 HiFiGAN 和 HN-SEP 模型）

## 环境配置（Conda）

> 如果尚未安装 Conda，请先从 [Miniconda 官网](https://www.anaconda.com/download) 下载并安装。

### 1. 创建 Conda 环境

```bash
# 创建 Python 3.12 环境
conda create -n hifiutau-engine python=3.12
conda activate hifiutau-engine
```

### 2. 安装依赖

#### ONNX 模式（CPU / DirectML）

```bash
# 安装 ONNX Runtime CPU 版（全平台通用）
pip install onnxruntime

# Windows DirectML 版（GPU 加速，可选）
pip install onnxruntime-directml

# 安装其他依赖
pip install -r requirements.txt
```

#### PyTorch 模式（CPU / CUDA）

> 请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择你的操作系统和 CUDA 版本，获取对应的安装命令。

```bash
# CPU 版
pip install torch

# CUDA 版（NVIDIA GPU，以下为 cu130 示例，请根据你的版本替换）
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 安装其他依赖
pip install -r requirements_pytorch.txt
```

### 3. 配置运行参数

编辑 `run_config.txt`：

```ini
# 仅对onnx模式生效
# device: cpu 或 dml（ONNX 模式）
device=cpu
# DML 工作进程数量（仅在 device=dml 时有效）
dml_workers=2
```

## 使用方式

### 1. 启动合成服务器

#### ONNX 模式（推荐，兼容性最好）

```bash
# CPU 模式 & DirectML 模式（Windows GPU）
python http_syn2_cpu.py
```

#### PyTorch 模式（需要 GPU，性能更优）

```bash
python http_syn2_pytorch.py
```

启动后服务器默认监听 `http://localhost:8000`，API 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/synthesize` | 接收 JSON 数据，返回 WAV 音频 |

### 2. 在 OpenUTAU 中使用

1. **下载专用 OpenUTAU**：[xiaobaijunya/OpenUtau-CustomRenderer](https://github.com/xiaobaijunya/OpenUtau-CustomRenderer)
2. **加载 UTAU 音源**
3. **选择合成引擎**：在音轨设置中选择 `CUSTOM_SERVER`
4. **等待合成**

> 提示：可以在 OpenUTAU 中调整预渲染的线程数。CPU 渲染建议 2 线程，GPU 渲染可根据显卡性能调高。
> 如果不希望修改参数时反复提交渲染，可以关闭预渲染，使用实时渲染模式。


## 支持的参数

### UTAU 传统参数

| 参数 | 名称 | 说明 |
|------|------|------|
| `VEL` | 发音速度 | 控制辅音长度 |
| `VOL` | 音量 | 整体音量控制 |
| `P` | 音量归一化 | NORM 模式 |
| `DYN` | 动态音量 | OpenUTAU 前端实现 |

### 曲线参数（自定义渲染器）

| 参数 | 名称 | 范围 | 说明 |
|------|------|------|------|
| `GWL` | 咆哮强度 | `0` ~ `100` | 控制嘶吼/咆哮效果，0=无效果，100=最大深度。颤音频率自动跟随音高 |
| `VOIC` | 发声程度 | `100` ~ `0` | 控制谐波/噪声比例。100=完全发声（纯谐波），0=纯气息。建议对硬且气声少的音源调至 70 左右 |
| `TENC` | 张力 | `-200` ~ `200` | 控制频谱倾斜。正值提亮（柔和），负值压暗（厚重） |
| `GENC` | 性别 | `-200` ~ `200` | 控制共振峰偏移（性别参数） |
| `BREC` | 气声 | `-100` ~ `100` | 控制气息噪声增益。-100=静音，0=原始，+100=×4 放大 |
| `BREL` | 低频气声 | `-100` ~ `100` | 2000Hz 以下低频气息噪声独立增益 |
| `BREH` | 高频气声 | `-100` ~ `100` | 2000Hz 以上高频气息噪声独立增益 |
| `PHTP` | 音量跟随 | `none` / `forward` / `backward` | 自适应音量控制。`none`=不处理，`forward`=跟随前音素音量，`backward`=跟随后音素音量。**目前仅 zhCVVC 和 zhCVV 音源支持自动设置**，如果音量爆炸建议改为 `none` |
| `STRT` | 拉伸模式 | `normal` / `loop` | `normal`=插值拉伸，`loop`=循环拉伸 |
| `SPLC` | 拼接模式 | `default` / `mel` | `default`=HiFiGAN 拼接（低精度，偶有偏移），`mel`=mel 域能量拼接（高精度，偏移更少） |
| `LOWC` | 音高低切 | `0` ~ `100` | 控制低切位置。值为 100 时可切到约第二谐波，跟随音高动态调整 |

## 项目结构

```
├── http_syn2.py                 # ONNX HTTP 服务入口（单进程）
├── http_syn2_cpu.py             # ONNX HTTP 服务（支持 CPU / DML Worker Pool）
├── http_syn2_pytorch.py         # PyTorch HTTP 服务入口
├── main_onnx.py                 # ONNX 合成入口 & 模型加载
├── main_pytorch.py              # PyTorch 合成入口 & 模型加载
├── run_config.txt               # 运行配置（device / dml_workers）
├── requirements.txt             # ONNX 模式依赖
├── requirements_pytorch.txt     # PyTorch 模式依赖
│
├── synthesis_pipeline/          # 合成管线（核心处理逻辑）
│   ├── engine.py                # 合成引擎（调度管线）
│   ├── fragment.py              # 音素片段处理（音频切割、mel 转换、时间拉伸）
│   ├── growl.py                 # 咆哮效果（时域延迟调制）
│   ├── tension_filter.py        # 动态张力滤波器（STFT 域频谱倾斜）
│   ├── post_process.py          # HN-SEP 后处理（breath/tension/voicing）
│   └── utils.py                 # 工具函数
│
├── tools/                       # 工具模块
│   ├── base_splicer.py          # 隐空间拼接基类
│   ├── base_hnsep.py            # HN-SEP 基类
│   ├── hidden_splicer.py        # ONNX HiddenSplicer（隐空间拼接）
│   ├── pytorch_splicer.py       # PyTorch HiddenSplicer
│   ├── hnsep_onnx.py            # HN-SEP ONNX 推理
│   ├── hnsep_pytorch.py         # HN-SEP PyTorch 推理
│   ├── nsf_hifigan.py           # HiFiGAN 模型定义（SplitGenerator）
│   ├── utils.py                 # 工具函数（AttrDict, init_weights）
│
├── hnsep_onnx/                  # HN-SEP ONNX 模型
├── hnsep/                       # HN-SEP PyTorch 模型
├── exported_onnx_v2/            # 导出的 ONNX 拆分模型
└── pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/  # HiFiGAN 模型 & 配置
```

## 技术说明

- **采样率**: 44100Hz
- **FFT 参数**: n_fft=2048, hop_size=512, win_size=2048
- **Mel 谱**: 128 频段, freq range 40~16000Hz
- **声码器**: PC-NSF-HiFiGAN（带基频条件的神经声码器）
- **谐波/噪声分离**: HN-SEP 模型用于独立控制气声、发声等参数

## 构建（GitHub Actions）

项目包含完整的 GitHub Actions CI/CD 配置（`.github/workflows/build.yml`），支持多平台自动打包：

| 平台 | 产物 |
|------|------|
| Windows x64 | `hifiserver-pytorch-win-x64.zip` |
| Linux x64 | `hifiserver-pytorch-linux-x64.zip` |
| macOS ARM64 | `hifiserver-pytorch-osx-arm64.zip` |

## 相关链接

- **专用 OpenUTAU（CustomRenderer 分支）**: [xiaobaijunya/OpenUtau-CustomRenderer](https://github.com/xiaobaijunya/OpenUtau-CustomRenderer)
- **声码器模型**: PC-NSF-HiFiGAN ([pc_nsf_hifigan_44.1k_hop512_128bin_2025.02](https://github.com/openvpi/SoundCodec))
- **HN-SEP**: [vocal-remover](https://github.com/yxlllc/vocal-remover/releases)

