from docutils import nodes
from docutils.parsers.rst.directives.admonitions import BaseAdmonition


class DefinitionDirective(BaseAdmonition):
    optional_arguments = 1  # optional custom title
    final_argument_whitespace = True
    node_class = nodes.admonition

    def run(self):
        self.options["class"] = self.options.get("class", []) + ["definition"]
        if not self.arguments:
            self.arguments = ["Definition"]  # default title
        return super().run()


def setup(app):
    app.add_directive("definition", DefinitionDirective)
    return {"parallel_read_safe": True}
