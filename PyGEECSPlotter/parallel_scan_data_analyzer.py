# Thread-parallel analyze_scan.
# Author: Claude
# Created: 2026-09-30

import concurrent.futures
import warnings

from tqdm import tqdm

from PyGEECSPlotter.scan_data_analysis import ScanDataAnalyzer, _aux_kwarg


class ParallelScanDataAnalyzer(ScanDataAnalyzer):
    """
    ``ScanDataAnalyzer`` with a thread-parallel ``analyze_scan``.

    Everything else (``load_scan_data``, ``filter_scan_data``, ``rebin``,
    ``display_scan``, ``mean_std_diagnostic``, ``aggregate_per_bin``, ...)
    is inherited unchanged and stays serial — only ``analyze_scan`` is
    overridden here.

    Per shot, ``load_data`` + ``analyze_data`` + ``write_analyzed_data`` /
    ``write_analyzed_lineouts`` run concurrently across shots in a
    ``ThreadPoolExecutor`` (each shot writes its own uniquely-named files).
    ``display_data`` / ``write_displayed_data`` (pyplot; slow, and a warning
    is issued if requested — use ``ScanDataAnalyzer`` for those) (and building
    ``add_columns_df``, merging it into ``self.data``, and the optional sfile
    write) still run on the main thread, one shot at a time, in the same
    ``active_data`` row order as the serial ``ScanDataAnalyzer`` — so output
    (row order, written files, merged columns) is identical to
    ``ScanDataAnalyzer.analyze_scan`` on the same inputs, just faster.

    This is safe **only** for an analyzer whose ``analyze_data`` /
    ``load_data``:

    - never mutate ``self.*`` in a way another concurrent call would see
      (a lazily-populated cache guarded by its own ``threading.Lock``, as in
      ``MagSpecAllEAnalyzer.default_background`` / ``EBeamProfileAnalyzer
      .default_background``, is fine — an unguarded lazy cache is not);
    - never touch ``matplotlib.pyplot`` (pyplot's global state machine is
      not thread-safe; that must stay confined to ``display_data`` /
      ``write_displayed_data``, which this class always runs single-threaded
      on the main thread);
    - don't wrap a stateful external resource with no documented
      thread-safety (e.g. ``WavefrontAnalyzer`` holds a live WaveKit
      ``HasoEngine`` and reused ``ComputePhaseSet`` objects that are
      mutated then used with no lock — **do not use ``WavefrontAnalyzer``
      with this class**; use the base ``ScanDataAnalyzer`` for it instead).

    If in doubt about a given analyzer, check `pygeecsplotter-dev`'s
    `architecture.md` ("Parallel analysis") or ask before assuming it's safe.

    Parameters
    ----------
    Same as ``ScanDataAnalyzer``.
    """

    def analyze_scan(self, analyzer,
        bg=None,
        display_data=False,
        write_columns_to_sfile=False,
        overwrite_columns=True,
        analysis_label='',
        extra_info_str='',
        write_analyzed=False,
        write_lineouts=False,
        write_displayed=False,
        close_displayed=True,
        max_workers=None,
        ):
        """
        Thread-parallel counterpart of ``ScanDataAnalyzer.analyze_scan``.

        Same parameters, behaviour and return value as
        ``ScanDataAnalyzer.analyze_scan``, plus:

        max_workers : int, optional
            Passed to ``concurrent.futures.ThreadPoolExecutor``. ``None``
            (default) uses Python's default (``min(32, os.cpu_count() + 4)``).

        See the class docstring for which analyzers this is safe to use with.
        """
        self.last_merged_columns = []
        rows = []
        analysis_dir = None
        self._warn_if_write_displayed_is_a_noop(display_data, write_displayed)
        if display_data or write_displayed:
            warnings.warn(
                "ParallelScanDataAnalyzer renders display_data / write_displayed figures "
                "one shot at a time on the main thread (pyplot is not thread-safe), so "
                "they will dominate the runtime. Use ScanDataAnalyzer.analyze_scan "
                "if you need them.", stacklevel=2)

        # Resolved (and created) once up front so the worker threads can write
        # per-shot files without racing to make the directory.
        if write_analyzed:
            analysis_dir = self.get_scan_data_analysis_dir(make_dir=True)

        def process(row):
            context, data, results, aux = self._process_row(analyzer, bg, row)
            if write_analyzed and data is not None:
                scan, shot_num = int(context['scan']), int(context['Shotnumber'])
                analyzer.write_analyzed_data(data, analysis_dir, scan, shot_num, context=context,
                                             **_aux_kwarg(analyzer.write_analyzed_data, aux))
                if write_lineouts:
                    analyzer.write_analyzed_lineouts(aux, analysis_dir, scan, shot_num)
            return context, data, results, aux

        for context, data, results, aux in self._iter_shots_parallel(
            analyzer, bg=bg, max_workers=max_workers, process=process
        ):
            rows.append({'scan': int(context['scan']), 'Shotnumber': int(context['Shotnumber']), **results})
            # write_analyzed / write_lineouts already done in the workers.
            analysis_dir = self._apply_shot_side_effects(
                analyzer, context, data, results, aux, analysis_dir,
                display_data=display_data, write_analyzed=False,
                write_lineouts=False, write_displayed=write_displayed,
                close_displayed=close_displayed,
            )

        return self._finish_analyze_scan(
            rows, analyzer, analysis_dir, write_columns_to_sfile=write_columns_to_sfile,
            analysis_label=analysis_label, extra_info_str=extra_info_str,
            overwrite_columns=overwrite_columns,
        )

    def _iter_shots_parallel(self, analyzer, bg=None, rows=None, max_workers=None,
                             show_progress=True, process=None):
        """
        Thread-parallel counterpart of ``_iter_shots``.

        Submits ``self._process_row(analyzer, bg, row)`` for every row up
        front to a ``ThreadPoolExecutor``, then yields the results in the
        same order ``_iter_shots`` would (i.e. ``rows``' original order —
        the row order of ``active_data``/``rows``, *not* completion order),
        so downstream output (``add_columns_df`` row order, per-shot
        display/write side effects) is identical to the serial path.

        Parameters, yields — same as ``_iter_shots``, plus:

        max_workers : int, optional
            Forwarded to ``ThreadPoolExecutor``.
        process : callable, optional
            ``process(row) -> (context, data, results, aux)`` run in the
            worker thread instead of ``_process_row`` (e.g. to also write
            the shot's analysed files there).
        """
        if process is None:
            def process(row):
                return self._process_row(analyzer, bg, row)
        if rows is None:
            rows = self.active_data
        # list(...) freezes row order up front; a DataFrame's row order can't
        # change under us mid-analysis since nothing here mutates self.data.
        row_list = [row for _, row in rows.iterrows()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, row) for row in row_list]

            # Yield in submission order as each result becomes ready, so the
            # caller's main-thread display/write work overlaps with the
            # workers still analysing later shots. The bar tracks shots
            # consumed by the caller (i.e. including its side effects).
            # .result() re-raises any exception from that shot's thread.
            progress = tqdm(total=len(futures)) if show_progress else None
            try:
                for future in futures:
                    result = future.result()
                    yield result
                    if progress is not None:
                        progress.update(1)
            finally:
                if progress is not None:
                    progress.close()
