"""CPU-provable core of the exact multi-position verifier (Path 1).

Spec: retroinfer-eval ``docs/superpowers/specs/2026-09-19-multiposition-verify-path1.md``.

Every module in this package is pure torch and must stay importable on a
machine without CUDA: no module here may import ``cache_engine`` (it compiles
``diff_offload`` at import, cache_engine.py:19-23), ``flash_cache_engine``,
``torch.utils.cpp_extension``, triton, or any attention extension. The tests in
retroinfer-eval ``tests/test_nosi_verify_core.py`` guard this at the source
level and load each module by file path, so the modules also do not import
each other.
"""
