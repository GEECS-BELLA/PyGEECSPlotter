"""Build the example notebooks from the example scripts.

The scripts in ``scripts/`` are the source of truth: plain Python with
``# %%`` cell markers (they run as-is, or cell by cell in VS Code / Spyder).
This writes an output-free ``.ipynb`` for each into ``notebooks/``:

    python Examples/build_notebooks.py            # all scripts
    python Examples/build_notebooks.py 02         # scripts whose name contains '02'

Cell markers:
    # %%              code cell
    # %% [markdown]   markdown cell; the following comment lines (``# text``)
                      become the markdown text
The module docstring becomes the first markdown cell.
"""

import ast
import pathlib
import sys

import nbformat

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = HERE / 'scripts'
NOTEBOOKS = HERE / 'notebooks'


def split_cells(text):
    """Yield (kind, source) for each ``# %%`` cell of a script."""
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    lines = text.splitlines()
    if doc:
        yield 'markdown', doc
        body_start = tree.body[0].end_lineno       # skip past the docstring
        lines = lines[body_start:]

    kind, buf = 'code', []

    def flush():
        src = '\n'.join(buf).strip('\n')
        if kind == 'markdown':
            src = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#')
                            for l in src.splitlines())
        return (kind, src) if src.strip() else None

    for line in lines:
        if line.startswith('# %%'):
            cell = flush()
            if cell:
                yield cell
            kind = 'markdown' if '[markdown]' in line else 'code'
            buf = []
        else:
            buf.append(line)
    cell = flush()
    if cell:
        yield cell


def build(script):
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    for kind, src in split_cells(script.read_text(encoding='utf-8')):
        make = nbformat.v4.new_markdown_cell if kind == 'markdown' else nbformat.v4.new_code_cell
        nb.cells.append(make(src))
    out = NOTEBOOKS / (script.stem + '.ipynb')
    nbformat.write(nb, str(out))
    return out, len(nb.cells)


def main(argv):
    NOTEBOOKS.mkdir(exist_ok=True)
    scripts = sorted(SCRIPTS.glob('*.py'))
    if argv:
        scripts = [s for s in scripts if any(a in s.name for a in argv)]
    for s in scripts:
        out, n = build(s)
        print(f'{s.name} -> notebooks/{out.name} ({n} cells)')


if __name__ == '__main__':
    main(sys.argv[1:])
