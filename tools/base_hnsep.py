"""
HN-SEP 谐波/噪声分离 — 公共基类

所有 HN-SEP（ONNX / PyTorch）共享的接口定义和工具方法。
具体差异只体现在模型加载和推理上（separate 方法）。
"""

import numpy as np


class BaseHnsep:
    """
    HN-SEP 谐波/噪声分离器基类。

    子类需实现:
      separate(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]
          输入波形 → (harmonic, noise)，均为 1D float32
    """

    def separate(self, waveform: np.ndarray) -> tuple:
        """分离音频为谐波和噪声分量。子类实现。"""
        raise NotImplementedError

    def __call__(self, waveform: np.ndarray) -> tuple:
        """兼容性调用，等价于 separate()。"""
        return self.separate(waveform)

    # ─── 输入标准化 ──────────────────────────────────────────

    @staticmethod
    def _ensure_1d(waveform: np.ndarray) -> np.ndarray:
        """确保波形为 1D float32 数组。"""
        if waveform.ndim == 2:
            waveform = waveform[0] if waveform.shape[0] == 1 else waveform[:, 0]
        return np.ascontiguousarray(waveform, dtype=np.float32)

    # ─── 后处理（breath / voicing / tension）────────────────

    def apply_breath_tension(
        self,
        waveform: np.ndarray,
        breath: float = 100.0,
        voicing: float = 100.0,
        tension: float = 0.0,
    ) -> np.ndarray:
        """
        使用 HN-SEP 对音频应用 breath（气声）、voicing（发声）、tension（张力）参数。

        原理:
          1. 使用 self.separate() 分离谐波和噪声
          2. breath 控制噪声（气息）电平
          3. voicing 控制谐波（发声）电平
          4. tension 控制谐波部分的频谱倾斜（预加重滤波）
          5. 最终音频 = 处理后的噪声 + 处理后的谐波

        Args:
            waveform: np.ndarray, shape (samples,) — 输入音频
            breath:   float, 气息量 (0-500, 默认 100 = 原始)
            voicing:  float, 发声量 (0-150, 默认 100 = 原始)
            tension:  float, 张力 (-100 ~ 100, 默认 0 = 原始)

        Returns:
            np.ndarray, shape (samples,) — 处理后的音频
        """
        # 如果参数都是默认值，直接返回
        if (abs(breath - 100) < 0.5 and abs(voicing - 100) < 0.5
                and abs(tension) < 0.5):
            return waveform

        harmonic, noise = self.separate(waveform)

        # 处理 breath（气息）
        breath = np.clip(breath, 0, 500)
        noise_out = noise * (breath / 100.0)

        # 处理 voicing（发声）
        voicing = np.clip(voicing, 0, 150)
        harmonic_scaled = harmonic * (voicing / 100.0)

        # 处理 tension（张力）
        tension = np.clip(tension, -100, 100)
        if abs(tension) > 0.5:
            harmonic_out = self._apply_tension_filter(harmonic_scaled, tension)
        else:
            harmonic_out = harmonic_scaled

        # 混合
        result = noise_out + harmonic_out

        # 保持原始响度水平
        original_rms = np.sqrt(np.mean(waveform ** 2))
        result_rms = np.sqrt(np.mean(result ** 2))
        if result_rms > 1e-8:
            result = result * (original_rms / result_rms)

        return result

    # ─── 张力滤波器 ──────────────────────────────────────────

    @staticmethod
    def _apply_tension_filter(waveform: np.ndarray, tension: float) -> np.ndarray:
        """
        对音频应用张力（频谱倾斜）滤波器。

        使用简单的一阶 FIR 滤波器:
          y[n] = x[n] - alpha * x[n-1]

        Args:
            waveform: np.ndarray, shape (samples,) — 输入音频
            tension:  float, -100 ~ 100

        Returns:
            np.ndarray, shape (samples,) — 滤波后的音频
        """
        alpha = tension / 100.0 * 0.97
        if abs(alpha) < 0.01:
            return waveform

        filtered = np.zeros_like(waveform)
        filtered[0] = waveform[0]
        for i in range(1, len(waveform)):
            filtered[i] = waveform[i] - alpha * waveform[i - 1]

        original_rms = np.sqrt(np.mean(waveform ** 2))
        filtered_rms = np.sqrt(np.mean(filtered ** 2))
        if filtered_rms > 1e-8:
            filtered = filtered * (original_rms / filtered_rms)

        return filtered
