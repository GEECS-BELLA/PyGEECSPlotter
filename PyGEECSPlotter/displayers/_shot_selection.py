"""
Shared shot picking for multi-diagnostic single-shot views
(``MultiDiagnosticAlignment``, ``ShotCompare``). Not part of the public
displayer API.
"""

import numpy as np


def most_present_positions(active, inputs):
    """Row positions in ``active`` with the most of ``inputs``'
    ``'<name> file_exists'`` flags set (every row if there are none)."""
    exists = [f'{name} file_exists' for name, _ in inputs
              if f'{name} file_exists' in active.columns]
    n_present = (active[exists] != 0).sum(axis=1).to_numpy() if exists \
        else np.zeros(len(active))
    return np.flatnonzero(n_present == n_present.max())


def select_position(active, inputs, selector):
    """
    Row position in ``active`` of one shot, or None.

    ``selector``: ``'first'`` / ``'last'`` (first / last shot with the most
    inputs present), an int ``Shotnumber``, or ``callable(active) -> int``
    row position.
    """
    if len(active) == 0:
        return None
    if isinstance(selector, (int, np.integer)):
        matches = np.flatnonzero(active['Shotnumber'].to_numpy() == int(selector))
        return int(matches[0]) if len(matches) else None
    if callable(selector):
        return int(selector(active))
    if selector not in ('first', 'last'):
        raise ValueError(f"Shot selector must be 'first', 'last', a Shotnumber or a "
                         f"callable, got {selector!r}.")
    best = most_present_positions(active, inputs)
    return int(best[-1] if selector == 'last' else best[0])


def select_positions(active, inputs, shots):
    """
    Row positions in ``active`` for several shots, in order, without
    duplicates.

    ``shots``: anything ``select_position`` takes; a list of those (e.g.
    Shotnumbers; ones not in ``active`` are skipped); ``'all'``; or
    ``('sample', n)`` — ``n`` shots evenly spaced through the active data.
    """
    if len(active) == 0:
        return []
    if isinstance(shots, str) and shots == 'all':
        return list(range(len(active)))
    if isinstance(shots, tuple) and len(shots) == 2 and shots[0] == 'sample':
        n = min(int(shots[1]), len(active))
        return sorted(set(np.linspace(0, len(active) - 1, n).round().astype(int).tolist()))
    if isinstance(shots, (list, np.ndarray)):
        positions = [select_position(active, inputs, s) for s in shots]
        return list(dict.fromkeys(p for p in positions if p is not None))
    pos = select_position(active, inputs, shots)
    return [] if pos is None else [pos]
