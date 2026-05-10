
import os
import numpy as np
import torch
import soundfile as sf
import io
from pathlib import Path
import onnxruntime

# 全局变量，用于存储预加载的ONNX模型会话
_global_ort_session = None

def set_global_ort_session(session):
    """设置全局ONNX模型会话"""
    global _global_ort_session
    _global_ort_session = session

def get_global_ort_session():
    """获取全局ONNX模型会话"""
    return _global_ort_session



def hifigan_synthesize(mel, output_wav_path,f0):
    """
    使用HifiGAN从mel缓存文件合成音频

    Args:
        cache_path: mel缓存文件路径(.npz)
        output_wav_path: 输出wav文件路径
        model_path: HifiGAN模型路径
        device: 计算设备('cuda'或'cpu')
    """

    default_f0 = 440.0

    # 使用预加载的ONNX模型会话
    global_ort_session = get_global_ort_session()
    if global_ort_session is None:
        raise RuntimeError("未找到预加载的ONNX模型，请确保HTTP服务已正确启动并预加载模型")

    print(f"使用预加载的ONNX模型")
    ort_session = global_ort_session

    # 准备输入数据
    mel_input = mel.astype(np.float32)
    mel_input = np.expand_dims(mel_input, axis=0).transpose(0, 2, 1)

    # 创建f0输入
    mel_frames = mel.shape[1]
    if f0 is not None:
        f0_input = np.zeros((1, mel_frames), dtype=np.float32)
        f0_len = min(len(f0), mel_frames)
        if f0_len > 0:
            f0_input[0, :f0_len] = f0[:f0_len].astype(np.float32)
            if f0_len < mel_frames:
                f0_input[0, f0_len:] = f0[f0_len-1]  # 复制最后一帧
        else:
            f0_input[0, :] = default_f0
    else:
        f0_input = np.full((1, mel_frames), default_f0, dtype=np.float32)
    print(f"F0 input shape: {f0_input.shape}, range: [{f0_input.min():.2f}, {f0_input.max():.2f}]")

    print(f"Synthesizing audio")
    input_data = {'mel': mel_input, 'f0': f0_input}
    output = ort_session.run(['waveform'], input_data)[0]
    audio = output[0]


    print(f"Saving audio to {output_wav_path}")
    if output_wav_path == None:
        # 将音频转换为二进制WAV格式
        buffer = io.BytesIO()
        sf.write(buffer, audio, 44100, 'PCM_16', format='WAV')
        buffer.seek(0)
        wav_bytes = buffer.getvalue()
        print(f"Done! Audio shape: {audio.shape}, WAV size: {len(wav_bytes)} bytes")
        return wav_bytes
    else:
        sf.write(output_wav_path, audio, 44100, 'PCM_16')
        print(f"Done! Audio shape: {audio.shape}")
        buffer = io.BytesIO()
        sf.write(buffer, audio, 44100, 'PCM_16', format='WAV')
        buffer.seek(0)
        wav_bytes = buffer.getvalue()
        print(f"Done! Audio shape: {audio.shape}, WAV size: {len(wav_bytes)} bytes")
        return wav_bytes



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Synthesize audio from mel cache using HifiGAN')
    parser.add_argument('--input', '-i', required=True, help='Input mel cache file path (.npz)')
    parser.add_argument('--output', '-o', required=True, help='Output WAV file path')
    parser.add_argument('--model', '-m', required=True, help='HifiGAN model path')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    hifigan_model_path = args.model
    try:
        from tools.preload_hifigan_model import preload_hifigan_model
        preload_hifigan_model(hifigan_model_path)
    except Exception as e:
        print(f"⚠️  模型预加载失败: {e}")

    hifigan_synthesize(args.input, args.output, args.model)