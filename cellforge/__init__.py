"""CellForge: agentic design of virtual cell models (Tang et al., arXiv:2508.02276).

Subpackages load lazily, so importing one stage does not import the others.
"""

from importlib import import_module

__version__ = "0.2.0"

_LAZY = {
    "Task_Analysis": ".Task_Analysis",
    "Method_Design": ".Method_Design",
    "Code_Generation": ".Code_Generation",
    "autorun": ".autorun",
    "retrieval": ".retrieval",
}


def __getattr__(name):
    if name in _LAZY:
        module = import_module(_LAZY[name], __name__)
    elif name == "LLMInterface":
        module = import_module(".llm", __name__).LLMInterface
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = module
    return module


__all__ = ["LLMInterface", *_LAZY]
