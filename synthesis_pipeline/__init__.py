"""
合成管线 — 与模型加载解耦的高性能语音合成模块。

使用方式:
    from synthesis_pipeline import SynthesisEngine

    engine = SynthesisEngine(splicer=my_splicer, hnsep_session=my_session)
    wav_bytes = engine.synthesize(json_data)
"""

from synthesis_pipeline.engine import SynthesisEngine

__all__ = ["SynthesisEngine"]
