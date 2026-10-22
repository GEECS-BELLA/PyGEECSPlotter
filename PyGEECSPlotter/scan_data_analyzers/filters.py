"""Named shot-selection rules for ScanDataAnalyzer.

A scan's selection is an ordered set of named rules, each re-evaluated
against the scan's current DataFrame. The selected rows are the rows every
rule keeps. Because the mask is rebuilt from the data rather than stored,
it can't fall out of step when columns are merged in, and a rule on
``temp Bin number`` follows ``rebin``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Filter:
    """One selection rule.

    kind : 'range'  keep low <= column <= high (either bound may be None)
           'values' keep column in values
           'rows'   keep where func(data) is True (func returns a bool Series
                    or array aligned with data)
    exclusive : invert the rule (keep what it would otherwise drop)
    file_rule : added automatically for missing diagnostic files; kept by
                ``clear_filters()`` unless asked otherwise
    """
    name: str
    kind: str
    column: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    values: Any = None
    func: Optional[Callable] = field(default=None, compare=False)
    exclusive: bool = False
    file_rule: bool = False

    def evaluate(self, data):
        """Boolean ndarray (len(data)) of the rows this rule keeps."""
        if self.kind == 'range':
            col = data[self.column]
            keep = np.ones(len(data), dtype=bool)
            if self.low is not None:
                keep &= (col >= self.low).to_numpy()
            if self.high is not None:
                keep &= (col <= self.high).to_numpy()
            keep &= col.notna().to_numpy()
        elif self.kind == 'values':
            keep = data[self.column].isin(np.asarray(self.values)).to_numpy()
        elif self.kind == 'rows':
            keep = np.asarray(self.func(data), dtype=bool)
            if keep.shape != (len(data),):
                raise ValueError(f"filter {self.name!r}: function returned shape {keep.shape}, "
                                 f"expected ({len(data)},)")
        else:
            raise ValueError(f"unknown filter kind {self.kind!r}")
        return ~keep if self.exclusive else keep

    def describe(self):
        if self.kind == 'range':
            if self.low is not None and self.high is not None:
                text = f'{self.low:g} <= {self.column} <= {self.high:g}'
            elif self.low is not None:
                text = f'{self.column} >= {self.low:g}'
            elif self.high is not None:
                text = f'{self.column} <= {self.high:g}'
            else:
                text = f'{self.column} not NaN'
        elif self.kind == 'values':
            vals = np.asarray(self.values)
            shown = ', '.join(str(v) for v in vals[:4]) + (', ...' if vals.size > 4 else '')
            text = f'{self.column} in [{shown}] ({vals.size} values)'
        else:
            text = getattr(self.func, '__name__', 'custom rule')
        return ('NOT ' if self.exclusive else '') + text


def combined_mask(filters, data):
    """AND of every rule's mask over ``data`` (all True when there are no rules)."""
    mask = np.ones(len(data), dtype=bool)
    for f in filters.values():
        mask &= f.evaluate(data)
    return mask


def summary(filters, data):
    """Multi-line description: each rule, how many rows it alone keeps, and the running total."""
    n = len(data)
    if not filters:
        return f'No filters: {n} / {n} shots.'
    lines = []
    running = np.ones(n, dtype=bool)
    for i, f in enumerate(filters.values(), 1):
        own = f.evaluate(data)
        running &= own
        tag = ' (files)' if f.file_rule else ''
        desc = f.describe()
        label = desc if f.name in (desc, f.column) else f'{f.name}: {desc}'
        lines.append(f'{i}. {label}{tag}  -> keeps {int(own.sum())}, '
                     f'{int(running.sum())} / {n} remain')
    return '\n'.join(lines)
