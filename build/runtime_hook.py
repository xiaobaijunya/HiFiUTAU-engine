"""PyInstaller runtime hook：预导入 scipy 模块，解决 lazy_loader 打包丢失问题。"""
import scipy.stats  # noqa: F401  # 含 _distn_infrastructure，必须最先加载
import scipy.signal  # noqa: F401  # librosa 依赖
import scipy.fft  # noqa: F401
import scipy.linalg  # noqa: F401
import scipy.interpolate  # noqa: F401
