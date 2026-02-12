from cotengra.oe import PathOptimizer

class HyperOptimizer(PathOptimizer):
    def __init__(
        self, minimize: str = "flops", reconf_opts: dict[None, None] | None = None, progbar: bool = False
    ) -> None: ...
