"""
隐空间混合拼接器 (ONNX)

继承 BaseSplicer，差异只在 ONNX Runtime 模型加载和推理。
公共流程（process/synthesize/splice）在 base_splicer.py 中。
"""

import numpy as np
import onnxruntime
from tools.base_splicer import BaseSplicer


class HiddenSplicer(BaseSplicer):
    """隐空间混合拼接器 — ONNX Runtime 版。"""

    def __init__(self, part1_onnx_path, part2_onnx_path, config_path,
                 device='dml', infer_threads=1):
        """
        Args:
            part1_onnx_path: part1.onnx 路径 (mel → 隐特征)
            part2_onnx_path: part2.onnx 路径 (隐特征 + f0 → 波形)
            config_path:     config.json 路径
            device:          'dml' (DirectML) 或 'cpu'
            infer_threads:   每个 session 的 intra_op 推理线程数（限制 CPU 占用）
        """
        super().__init__(config_path)

        providers = self._resolve_providers(device)
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = max(1, int(infer_threads))
        self.part1_session = onnxruntime.InferenceSession(
            part1_onnx_path, providers=providers, sess_options=so)
        self.part2_session = onnxruntime.InferenceSession(
            part2_onnx_path, providers=providers, sess_options=so)
        print(f'[HiddenSplicer] ONNX 模型已加载, providers={providers}, '
              f'intra_op_threads={so.intra_op_num_threads}')

    # ─── 抽象方法实现 ────────────────────────────────────────

    def part1_encode(self, mel_2d: np.ndarray) -> np.ndarray:
        """mel (128, T) → ONNX part1 → 隐特征 (1, 128, T*64)。"""
        mel_input = np.expand_dims(mel_2d.astype(np.float32), axis=0)
        return self.part1_session.run(['feat'], {'mel': mel_input})[0]

    def part2_synthesize(self, feat: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """隐特征 + f0 → ONNX part2 → 波形。"""
        f0_input = f0.reshape(1, -1).astype(np.float32)
        feat_input = feat.astype(np.float32)
        wav = self.part2_session.run(
            ['waveform'], {'feat': feat_input, 'f0': f0_input}
        )[0]
        return wav.squeeze()

    # ─── provider 解析 ──────────────────────────────────────

    @staticmethod
    def _resolve_providers(device: str) -> list:
        """根据 device 名称解析 ONNX Runtime provider 列表。"""
        device = device.lower()
        if device == 'cpu':
            return ['CPUExecutionProvider']
        if device in ('dml', 'directml'):
            return ['DmlExecutionProvider', 'CPUExecutionProvider']
        raise ValueError(f'未知 device="{device}"，仅支持 cpu 或 dml')
