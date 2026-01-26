import os
import subprocess
import sys

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class IncludeCliHelp(SphinxDirective):
    """
    Implements ``.. include-cli-help:: <path/to/prog>``
    """

    has_content = False
    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        document = self.state.document
        rel_path, path = self.env.relfn2path(self.arguments[0].strip())
        if os.path.isfile(path):
            self.state.document.settings.record_dependencies.add(path)
        else:
            self.severe(f"can't locate ``{path}``")

        if path.endswith(".py"):
            cmd = [sys.executable]
        else:
            cmd = []
        cmd.extend([str(path), "--help"])
        rslt = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if rslt.returncode == 0:
            content = rslt.stdout.decode()
        else:
            msg = f"there was a problem executing ``{' '.join(cmd)}``"
            document.reporter.warning(msg, line=self.lineno)
            content = f"<{msg}>"
        content = content.strip()
        literal = nodes.literal_block(content, content)
        return [literal]


def setup(app):
    app.add_directive("include-cli-help", IncludeCliHelp)

    return {"version": "0.1"}  # identifies the version of our extension
