# The slow end-of-scan work of a live job, in a separate process, so the
# watcher is ready for the next scan as soon as the sfile is merged:
#
#   python -m PyGEECSPlotter.scan_data_analyzers.live_end <spec.pkl>
#
# The spec (written by LiveShotWatcher, end_process=True) holds the LiveJob,
# the sfile, the job's results and the output folder. The process loads the
# scan with the job's columns (from the sfile if they were merged, else from
# the results in the spec), runs the job's displayers, saving their figures, and,
# if the job asks for write_displayed, draws every shot's display_data figure.
# Output goes to analysis/ScanNNN/online/end_of_scan.log.

import os
import pickle
import sys
import time


def load_scan(spec):
    """The scan (sfile) with the job's result columns."""
    from PyGEECSPlotter.scan_data_analyzers.scan_data_analysis import ScanDataAnalyzer
    from PyGEECSPlotter.scan_data_analyzers.sfile_frame import merge_results, rename_result_columns
    job = spec['job']
    scan = ScanDataAnalyzer(sfilename=spec['sfile'])
    scan.load_scan_data()
    results = spec.get('results')           # the job's per-shot results, when not merged into the sfile
    if not spec.get('merged') and results is not None and len(results):
        results = results.drop_duplicates(['scan', 'Shotnumber'], keep='last')
        scan.data = merge_results(scan.data, rename_result_columns(results, job.column_prefix,
                                                                   job.analysis_label or ''))
    return scan


def end_of_scan(spec):
    """Run the job's displayers (and write_displayed) on the finished scan."""
    import matplotlib.pyplot as plt
    from PyGEECSPlotter.scan_data_analyzers.live_analysis import run_displayers
    job = spec['job']
    t0 = time.monotonic()
    scan = load_scan(spec)
    if scan.aborted:
        print(f'{spec["sfile"]}: no shots')
        return
    if job.displayers:
        with scan.filtered():
            job.analyzer.register_with_scan(scan)
            run_displayers(scan, job.displayers, spec['output_dir'], registered=job.analyzer)
        plt.close('all')
        print(f'{len(job.displayers)} scan views in {spec["output_dir"]} ({time.monotonic() - t0:.1f} s)')
    if job.write_displayed:
        t1 = time.monotonic()
        with scan.filtered():
            job.analyzer.register_with_scan(scan)
            scan.analyze_scan(job.analyzer, bg=job.bg(scan), display_data=True, write_displayed=True,
                              write_analyzed=False, parallel=False, write_columns_to_sfile=False,
                              analysis_label=job.analysis_label or '', column_prefix=job.column_prefix)
        plt.close('all')
        print(f'per-shot figures written ({time.monotonic() - t1:.1f} s)')


def main(path):
    import matplotlib
    matplotlib.use('Agg')
    with open(path, 'rb') as f:
        spec = pickle.load(f)
    end_of_scan(spec)
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == '__main__':
    main(sys.argv[1])
