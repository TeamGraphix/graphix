from typing import Optional
from graphix.device_interface import DeviceBackend
from graphix.pattern import Pattern


def get_backend(name: str, pattern: Optional[Pattern] = None, **kwargs) -> DeviceBackend:
    if name == "ibmq":
        try:
            from graphix_ibmq.backend import IBMQBackend
        except ImportError:
            raise ImportError("Please install graphix-ibmq via `pip install graphix-ibmq`.")

        backend = IBMQBackend(**kwargs)
        if pattern is not None:
            backend.set_pattern(pattern)
        return backend
    else:
        raise ValueError(f"Unknown backend: {name}")