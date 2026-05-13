"""Non-intrusive diagnostic probes for Mamba-3 internals."""

__all__ = ["BCCosineProbe", "BCProbe", "bc_cosine_summary"]


def __getattr__(name):
    if name in {"BCCosineProbe", "bc_cosine_summary"}:
        from .bc_cosine import BCCosineProbe, bc_cosine_summary

        return {"BCCosineProbe": BCCosineProbe, "bc_cosine_summary": bc_cosine_summary}[name]
    if name == "BCProbe":
        from .probe_bc import BCProbe

        return BCProbe
    raise AttributeError(name)
