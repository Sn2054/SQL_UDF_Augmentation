"""Figures for the lab PC runtime measurement (profile_udf_queries.py, runs 1 and 2). CPU only, reads spreadsheets.

    CUDA_VISIBLE_DEVICES="" python plot_labpc_runtime.py \
        --profile_dir results/udf_query_profile_labpc --out_dir results/udf_query_profile_labpc/figures_all
    (add --bare --out_dir .../figures_all_slides for the slide versions)

Since 2026-11-20 the sheets cover every test query (sample 2026-11-12 to 11-14, the remaining push-down queries
2026-11-18 to 11-20, resumed into the same files) and already carry both models' predictions from the best
checkpoints (labpc_results/predictions_all), so no prediction override is needed. Each run has one clock log per
timing session; figure 2 draws them one after the other.

Run 1 (graceful_threads12/): GRACEFUL's procedure, 12 threads -> figure 3 (our time vs GRACEFUL's label).
Run 2 (passes_threads1/):    1 thread, 3 passes, real seconds -> every other figure.
Every figure covers employee, financial, fhnk and seznam, push-down and pull-up, and prints n per group.

  1_measurement_passes      spread of a query's 3 passes (coefficient of variation)
  2_clock                   CPU clock over both runs (Windows-side log; WSL exposes no temperature)
  3a_runtime_vs_graceful    mean time per query, GRACEFUL's label vs ours (run 1)
  3b_runtime_vs_graceful_per_query  one dot per query, ours vs the label (run 1)
  4a_q50_by_query_time      q50 baseline vs ours by total query time
  4b_q50_by_udf_time        the same by UDF time, with a < 1 ms bin
  5_q50_by_udf_type         q50 by UDF position (WHERE / SELECT) and control flow
  6a_udf_vs_rest_of_query   UDF time vs the rest of the query, one dot per query
  6b_udf_vs_total_time      UDF time vs total query time
  7_qerror_vs_udf_time      one dot per query: UDF time vs q-error, baseline and ours, bin medians and n
  7_qerror_vs_udf_time_<db> the same, one database per figure
  8_low_runtime_udfs        UDFs under 1 ms: DuckDB's estimated vs actual rows into the UDF
  9_runtime_distributions   UDF time, total time and GRACEFUL's label per database
  low_runtime_udfs.csv      every query whose UDF takes < 1 ms (run 2)
  summary.csv               the numbers behind the titles
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

DBS = ['employee', 'financial', 'fhnk', 'seznam']
WORKLOADS = {'pushdown': 'push-down', 'pullup': 'pull-up'}
OURS, BASE, GRACEFUL = '#2a78d6', '#98a6b3', '#eb6834'
INK, INK2, GRID, SURFACE, NAVY = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#26326c'
BINS = [0, 0.001, 0.1, 0.5, 2, 10, np.inf]
LABELS = ['<1 ms', '1-100 ms', '0.1-0.5 s', '0.5-2 s', '2-10 s', '>10 s']
MIN_N = 20   # groups with fewer queries are drawn faded and marked
RUN1, RUN2 = 'graceful_threads12', 'passes_threads1'
OVERRIDDEN = set()
BARE = False
DROP_SMALL = False   # slide versions leave out groups with fewer than MIN_N queries
SAMPLE = ('Queries: every push-down and pull-up test query of each database. '
          'q-error = max(pred / label, label / pred) against GRACEFUL\'s recorded runtime. Augmented: our best hybrid + '
          'SELU checkpoint per database.')
MACHINE = 'Lab PC: AMD Ryzen 5 7500F (1 socket, 6 cores / 12 threads), clock capped at ~2.41 GHz, boost off.'
CAPPED_MHZ = 2413

FONT = '/mnt/data/shehryar/tools/libreoffice/root/opt/libreoffice26.2/share/fonts/truetype/LiberationSerif-Regular.ttf'


def style():
    family = 'serif'
    if os.path.exists(FONT):
        for f in ['Regular', 'Bold', 'Italic']:
            font_manager.fontManager.addfont(FONT.replace('Regular', f))
        family = 'Liberation Serif'   # metric-compatible with Times New Roman (the deck font)
    plt.rcParams.update({
        'font.family': family, 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.axisbelow': True, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'legend.frameon': False,
    })


def load(profile_dir, run, ours_override=None):
    """{(db, workload): per-query frame with both models' q-errors}.

    ours_override: a predictions folder (<db>/<db>_<workload>_pred_augmented.csv, from profile_udf_queries.py
    --step predict) whose augmented predictions replace the sheet's, for databases whose best checkpoint is newer
    than the one the lab PC used. Predictions do not depend on the machine."""
    q = {}
    for db in DBS:
        for wl in WORKLOADS:
            b = pd.read_excel(f'{profile_dir}/{run}/{db}/{db}_baseline.xlsx', sheet_name=wl)
            a = pd.read_excel(f'{profile_dir}/{run}/{db}/{db}_augmented.xlsx', sheet_name=wl)
            a = a[['query_id', 'predicted_runtime_s', 'q_error']]
            f = ours_override and f'{ours_override}/{db}/{db}_{wl}_pred_augmented.csv'
            if f and os.path.exists(f):
                n = pd.read_csv(f)
                chk = n.merge(b[['query_id', 'graceful_runtime_s']], on='query_id')
                assert len(chk) == len(b) == len(n) and np.allclose(chk.graceful_runtime_s_x, chk.graceful_runtime_s_y), f
                a = n[['query_id', 'predicted_runtime_s', 'q_error']]
                OVERRIDDEN.add(db)
            d = b.merge(a, on='query_id', suffixes=('_base', '_ours'))
            # a query is usable when at least one timed run succeeded (one seznam query in run 2 failed 2 of its 3
            # passes with a UDF ZeroDivisionError that depends on DuckDB's run-time filter order; its time is pass 3's)
            part = d[(d.status != 'ok')]
            assert len(d) == len(b) == len(a) and (part.runs_ok >= 1).all() and d.total_time_s.notna().all(), (run, db, wl)
            for _, r in part.iterrows():
                print(f'{run} {db} {wl}: {r.query_id} timed in {r.runs_ok} of {r.runs_total} runs ({r.status[:60]}...)')
            q[db, wl] = d
    return q


def load_runs(profile_dir, run):
    return {(db, wl): pd.read_csv(f'{profile_dir}/{run}/{db}/{db}_{wl}_runs.csv') for db in DBS for wl in WORKLOADS}


def load_clock(profile_dir, run):
    """One frame per timing session (clock_<start>.csv), in time order."""
    out = []
    for f in sorted(glob.glob(f'{profile_dir}/{run}/clock_*.csv')):
        c = pd.read_csv(f)
        c.columns = ['time', 'perf_pct', 'freq_mhz']
        c = c[(c.perf_pct >= 0) & (c.freq_mhz >= 0)]          # typeperf writes -1 when it has no sample
        c['time'] = pd.to_datetime(c.time, format='%m/%d/%Y %H:%M:%S.%f')
        c['mhz'] = c.freq_mhz * c.perf_pct / 100
        out.append(c)
    return out


def save(fig, out_dir, name):
    for ext in ['png', 'pdf']:
        fig.savefig(f'{out_dir}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def title(fig, text, note=None, note_y=-0.02):
    if BARE:      # for slides: the slide carries the title and the note
        return
    fig.suptitle(text, x=0.01, ha='left', fontsize=13.5, color=NAVY)
    if note:
        fig.text(0.01, note_y, note, fontsize=9.5, color=INK2, va='top', wrap=True)


def grid8(figsize=(16, 7.4), **kw):
    """2 rows (push-down, pull-up) x 4 columns (databases)."""
    fig, axes = plt.subplots(2, 4, figsize=figsize, **kw)
    cells = [((db, wl), axes[i, j]) for i, wl in enumerate(WORKLOADS) for j, db in enumerate(DBS)]
    return fig, axes, cells


def figlegend(fig, *args, **kw):
    """Figure legend; in the slide versions (no title row) it goes under the panels instead of the top right."""
    if BARE:
        kw.update(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=max(kw.get('ncol', 1), 2))
    return fig.legend(*args, **kw)


def gain(base, ours):
    return 100 * (base - ours) / base


def bars(ax, labels, base, ours, ns):
    """Grouped bars baseline vs ours, % change above each pair, n under each label.

    Groups under MIN_N queries are faded and starred, and do not set the y-axis: a bar taller than the axis is cut at
    the top and its two q50 values are printed on it."""
    x = np.arange(len(labels))
    w = 0.38
    ok = [i for i in range(len(labels)) if ns[i] >= MIN_N] or list(range(len(labels)))
    top = max(max(base[i], ours[i]) for i in ok) * 1.12
    cap = top * 1.08
    for i in range(len(labels)):
        small = ns[i] < MIN_N
        alpha = 0.35 if small else 1
        hb, ho = min(base[i], cap), min(ours[i], cap)
        ax.bar(x[i] - w / 2 - 0.01, hb, w, color=BASE, alpha=alpha, label='Baseline' if i == 0 else None)
        ax.bar(x[i] + w / 2 + 0.01, ho, w, color=OURS, alpha=alpha, label='Augmented' if i == 0 else None)
        g = round(gain(base[i], ours[i])) + 0          # + 0 turns -0 into 0
        txt = f'{g:+.0f}%' + ('*' if small else '')
        if max(base[i], ours[i]) > cap:
            txt += f'\n({base[i]:.1f} vs {ours[i]:.1f})'
        ax.text(x[i], max(hb, ho) + top * 0.02, txt, ha='center', va='bottom', fontsize=9, color=INK2 if small else INK,
                fontweight='bold' if g > 0 and not small else 'normal')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\nn={n}' for l, n in zip(labels, ns)], fontsize=8.5)
    ax.set_ylim(1, cap * 1.18)    # q-error starts at 1
    ax.grid(axis='x', visible=False)


def binned(d, col):
    d = d.assign(bucket=pd.cut(d[col], BINS, labels=LABELS, include_lowest=True, right=False))
    g = d.groupby('bucket', observed=True).agg(n=('q_error_base', 'size'), base=('q_error_base', 'median'),
                                               ours=('q_error_ours', 'median'), x=(col, 'median'))
    return g[g.n >= MIN_N] if DROP_SMALL else g


# ---------------------------------------------------------------- figures

def fig_passes(q, out):
    fig, axes, cells = grid8((16, 6.4), sharex=True, sharey=True)
    for (db, wl), ax in cells:
        cv = q[db, wl].total_time_cv.dropna().clip(lower=1e-4) * 100     # NaN: a query with only one successful pass
        ax.hist(cv, bins=np.logspace(-2, 2, 41), color=OURS, edgecolor=SURFACE, linewidth=0.6,
                weights=np.full(len(cv), 100 / len(cv)))
        ax.set_xscale('log')
        ax.set_xticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])
        ax.axvline(cv.median(), color=INK, linewidth=1.2, linestyle='--')
        ax.set_title(f'{db} {WORKLOADS[wl]} (n={len(cv):,})', loc='left', fontsize=11)
        ax.text(0.97, 0.95, f'median {cv.median():.2f}%', transform=ax.transAxes, ha='right', va='top', fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("spread of a query's 3 runs (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel('% of queries')
    allcv = pd.concat([d.total_time_cv.dropna() for d in q.values()])
    title(fig, f'Each query was run 3 times (in order, reversed, random): median spread {100 * allcv.median():.2f}%',
          "x: standard deviation of a query's 3 total times as a % of their mean (e.g. 1.00, 1.01 and 0.99 s give "
          'about 1%). 1 DuckDB thread, pinned to one CPU. ' + MACHINE)
    fig.tight_layout()
    save(fig, out, '1_measurement_passes')


def fig_clock(profile_dir, runs1, runs2, out):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2))
    for ax, run, runs, name in [(axes[0], RUN1, runs1, 'Run 1: GRACEFUL\'s procedure, 12 threads'),
                                (axes[1], RUN2, runs2, 'Run 2: our procedure, 1 thread, 3 passes')]:
        sessions = load_clock(profile_dir, run)
        offset, i = 0.0, 0
        for s_no, c in enumerate(sessions):
            t0 = c.time.min()
            h = offset + (c.time - t0).dt.total_seconds() / 3600
            ax.plot(h, c.mhz / 1000, color=OURS, linewidth=0.5)
            if s_no:
                ax.axvline(offset, color=INK2, linewidth=1.2, linestyle='--')
                ax.text(offset + 0.15, 3.25, f'resumed {t0:%m-%d %H:%M}', fontsize=8.5, color=INK2, va='top')
            # where each database / workload starts in this session (runs CSV: Unix epochs; clock log: KST)
            e0 = pd.Timestamp(t0).tz_localize('Asia/Seoul').timestamp()
            e1 = pd.Timestamp(c.time.max()).tz_localize('Asia/Seoul').timestamp()
            starts = sorted((e[(e >= e0) & (e <= e1)].min(), k) for k, r in runs.items()
                            for e in [r.start_epoch] if ((e >= e0) & (e <= e1)).any())
            for e, (db, wl) in starts:
                x = offset + (e - e0) / 3600
                ax.axvline(x, color=GRID, linewidth=1.5, zorder=0)
                ax.text(x + 0.05, [0.15, 0.65, 1.15][i % 3], f'{db}\n{WORKLOADS[wl]}', fontsize=8, color=INK2,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor=SURFACE, edgecolor='none', alpha=0.85))
                i += 1
            offset = h.max()
        c = pd.concat(sessions)
        ax.set_xlim(0, offset)
        ax.set_ylim(0, 4)
        ax.set_ylabel('clock (GHz)')
        ax.axhline(3.7, color=INK2, linewidth=1, linestyle=':')
        ax.text(offset, 3.6, 'base clock 3.7 GHz (uncapped)', ha='right', va='top', fontsize=9, color=INK2)
        ax.set_title(f'{name}: median {c.mhz.median() / 1000:.2f} GHz over {len(c):,} one-second samples '
                     f'({len(sessions)} sessions)', loc='left', fontsize=11)
    axes[1].set_xlabel('hours of measurement (sessions drawn one after the other)')
    title(fig, 'The CPU clock stayed at the ~2.41 GHz cap through both runs',
          'Clock = Windows "Processor Frequency" x "% Processor Performance" / 100, logged every second on the '
          'Windows host (WSL cannot read the clock or the temperature). Cap: Windows maximum processor state 54%, '
          'boost off (the lowest Windows reaches on this CPU). ' + MACHINE)
    fig.tight_layout()
    save(fig, out, '2_clock')


def fig_runtime(q1, out, rows):
    keys = [(db, wl) for wl in WORKLOADS for db in DBS]
    labels = [f'{db}\n{WORKLOADS[wl]}\nn={len(q1[db, wl])}' for db, wl in keys]
    g = [q1[k].graceful_runtime_s.mean() for k in keys]
    o = [q1[k].total_time_s.mean() for k in keys]
    ratio = [(q1[k].total_time_s / q1[k].graceful_runtime_s).median() for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.bar(x - w / 2 - 0.01, g, w, color=GRACEFUL, label="GRACEFUL's recorded runtime (label)")
    ax.bar(x + w / 2 + 0.01, o, w, color=OURS, label="Lab PC: GRACEFUL's procedure, 12 threads")
    top = max(g)
    for i in range(len(x)):
        ax.text(x[i] - w / 2, g[i] + top * 0.015, f'{g[i]:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(x[i] + w / 2, o[i] + top * 0.015, f'{o[i]:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(x[i], top * 1.12, f'×{ratio[i]:.2f}', ha='center', fontsize=10.5, color=INK)
    ax.set_title('× = lab PC / label, median over the queries', loc='left', fontsize=10, color=INK2)
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel('mean time per query (s)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.93))
    title(fig, f'Same queries with GRACEFUL\'s procedure: our times are {min(ratio):.2f}-{max(ratio):.2f}× their labels',
          "Lab PC: run 1, GRACEFUL's procedure (copy + VACUUM ANALYZE, default threads = 12, file order, per repetition a "
          'hidden run then a timed EXPLAIN ANALYZE; mean of the 2 timed repetitions, as GRACEFUL builds its label). '
          "Still different from GRACEFUL's machine (CloudLab c8220: 2 x Xeon E5-2660 v2, 2 x 10 cores, 2.2 GHz, hard "
          'disks, 256 GB): CPU model and generation, 1 vs 2 sockets, 12 vs 20 threads, SSD, 16 GB RAM, WSL 2. '
          + MACHINE, note_y=-0.03)
    fig.tight_layout()
    save(fig, out, '3a_runtime_vs_graceful')
    for k, r in zip(keys, ratio):
        rows.append({'figure': '3a', 'db': k[0], 'workload': k[1], 'n': len(q1[k]), 'mean_label_s': q1[k].graceful_runtime_s.mean(),
                     'mean_ours_s': q1[k].total_time_s.mean(), 'median_ratio_ours_to_label': r})

    fig, axes, cells = grid8((16, 7.6), sharex=True, sharey=True)
    lim = [1e-2, 100]
    for (db, wl), ax in cells:
        d = q1[db, wl]
        ax.scatter(d.graceful_runtime_s, d.total_time_s, s=5, color=OURS, alpha=0.35, linewidths=0)
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=':')
        r = (d.total_time_s / d.graceful_runtime_s).median()
        ax.plot(lim, [l * r for l in lim], color=GRACEFUL, linewidth=1.2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim)
        rho = d.total_time_s.rank().corr(d.graceful_runtime_s.rank())
        ax.set_title(f'{db} {WORKLOADS[wl]} (n={len(d):,})\nrank corr. {rho:.3f}, median ratio {r:.2f}',
                     loc='left', fontsize=10.5)
        rows.append({'figure': '3b', 'db': db, 'workload': wl, 'n': len(d), 'spearman_ours_vs_label': rho})
    for ax in axes[1]:
        ax.set_xlabel("GRACEFUL's label (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel('lab PC, run 1 (s)')
    title(fig, "One dot per query: our time follows GRACEFUL's label at a near-constant factor",
          'Dotted: lab PC = label. Orange: lab PC = median ratio x label. Rank correlation = Spearman. Run 1 (GRACEFUL\'s '
          'procedure, 12 threads). ' + MACHINE)
    fig.tight_layout()
    save(fig, out, '3b_runtime_vs_graceful_per_query')


def fig_binned(q, out, col, name, head, note, rows):
    fig, axes, cells = grid8((17, 8.2))
    for (db, wl), ax in cells:
        g = binned(q[db, wl], col)
        bars(ax, list(g.index), g.base.values, g.ours.values, g.n.values)
        ax.set_title(f'{db} {WORKLOADS[wl]} (n={len(q[db, wl]):,})', loc='left', fontsize=11)
        for b, r in g.iterrows():
            rows.append({'figure': name[:2], 'db': db, 'workload': wl, 'bin': b, 'n': r.n, 'q50_base': r.base,
                         'q50_ours': r.ours, 'gain_pct': gain(r.base, r.ours)})
    for ax in axes[:, 0]:
        ax.set_ylabel('q50 (median q-error)')
    figlegend(fig, *axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=2, bbox_to_anchor=(1, 1.0))
    title(fig, head, f'{note} % = change of q50 vs the baseline (positive = augmented better). * and faded: fewer than '
          f'{MIN_N} queries, too few to read. Run 2 (1 thread). {SAMPLE}')
    fig.tight_layout()
    save(fig, out, name)


DB_COLORS = {'employee': '#2a78d6', 'financial': '#eb6834', 'fhnk': '#3a9a5b', 'seznam': '#7b4fb0'}


def fig_gain_trend(q, out, rows):
    """q50 improvement (%) per runtime bin, every database on one axis; only bins with >= MIN_N queries."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharey=True)
    for i, (col, what) in enumerate([('total_time_s', 'total query time (lab PC)'), ('udf_time_s', 'UDF time (lab PC)')]):
        for j, wl in enumerate(WORKLOADS):
            ax = axes[i, j]
            for db in DBS:
                d = q[db, wl].assign(bucket=lambda x: pd.cut(x[col], BINS, labels=LABELS, include_lowest=True,
                                                             right=False))
                g = d.groupby('bucket', observed=False).agg(n=('q_error_base', 'size'), base=('q_error_base', 'median'),
                                                            ours=('q_error_ours', 'median'))
                g = g[g.n >= MIN_N]
                xs = [LABELS.index(b) for b in g.index]
                ys = gain(g.base, g.ours).values
                ax.plot(xs, ys, marker='o', markersize=6, linewidth=1.8, color=DB_COLORS[db],
                        label=db if (i, j) == (0, 0) else None)
                for x_, y_, n_ in zip(xs, ys, g.n.values):
                    rows.append({'figure': '4c', 'db': db, 'workload': wl, 'bin': f'{what}: {LABELS[x_]}', 'n': n_,
                                 'gain_pct': y_})
            ax.axhline(0, color=INK2, linewidth=1)
            ax.set_xticks(range(len(LABELS)))
            ax.set_xticklabels(LABELS, fontsize=10)
            ax.set_xlim(-0.4, len(LABELS) - 0.6)
            ax.set_title(f'{WORKLOADS[wl]}, bins by {what}', loc='left', fontsize=11.5)
            ax.grid(axis='x', visible=False)
        axes[i, 0].set_ylabel('q50 improvement over baseline (%)')
    figlegend(fig, *axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=4)
    title(fig, 'q50 improvement per runtime bin, all four databases on one axis',
          f'Positive = augmented better than the baseline in that bin. Only bins with at least {MIN_N} queries. Bins '
          "by the lab PC's run-2 times; the q-errors are against GRACEFUL's labels. " + SAMPLE)
    fig.tight_layout()
    save(fig, out, '4c_gain_by_runtime_bin')


def fig_udf_type(q, out, rows):
    fig, axes, cells = grid8((17, 8.2))
    order = [f'{p}\n{c}' for p in ['WHERE', 'SELECT'] for c in ['plain', 'loop', 'branch']]
    for (db, wl), ax in cells:
        d = q[db, wl]
        pos = d.udf_pos_in_query.map(lambda p: 'SELECT' if p == 'select' else 'WHERE')
        ctrl = np.where(d.udf_num_loops > 0, 'loop', np.where(d.udf_num_branches > 0, 'branch', 'plain'))
        d = d.assign(group=pos + '\n' + ctrl)
        g = d.groupby('group').agg(n=('q_error_base', 'size'), base=('q_error_base', 'median'),
                                   ours=('q_error_ours', 'median')).reindex(order).dropna()
        if DROP_SMALL:
            g = g[g.n >= MIN_N]
        bars(ax, list(g.index), g.base.values, g.ours.values, g.n.astype(int).values)
        ax.set_title(f'{db} {WORKLOADS[wl]} (n={len(q[db, wl]):,})', loc='left', fontsize=11)
        for b, r in g.iterrows():
            rows.append({'figure': '5', 'db': db, 'workload': wl, 'bin': b.replace('\n', ' '), 'n': int(r.n),
                         'q50_base': r.base, 'q50_ours': r.ours, 'gain_pct': gain(r.base, r.ours)})
    for ax in axes[:, 0]:
        ax.set_ylabel('q50 (median q-error)')
    figlegend(fig, *axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=2, bbox_to_anchor=(1, 1.0))
    title(fig, 'q50 by where the UDF sits and its control flow',
          'WHERE / SELECT: the UDF\'s place in the query; plain: no loop or branch; loop: at least one loop; branch: '
          f'a branch but no loop. Pull-up has the UDF in WHERE only. % = change of q50 vs the baseline. * and faded: '
          f'fewer than {MIN_N} queries. {SAMPLE}')
    fig.tight_layout()
    save(fig, out, '5_q50_by_udf_type')


def fig_udf_sql(q, out, x, rows):
    fig, axes, cells = grid8((16, 7.6), sharex=True, sharey=True)
    lim = [1e-5, 100]
    for (db, wl), ax in cells:
        d = q[db, wl]
        other = (d.sql_time_s if x == 'rest' else d.total_time_s).clip(lower=lim[0])
        udf = d.udf_time_s.clip(lower=lim[0])
        low = d.udf_time_s < 0.001
        ax.scatter(other[~low], udf[~low], s=5, color=OURS, alpha=0.35, linewidths=0)
        ax.scatter(other[low], udf[low], s=14, color=GRACEFUL, marker='x', linewidths=0.9,
                   label='UDF < 1 ms' if (db, wl) == ('employee', 'pushdown') else None)
        ax.axhline(1e-3, color=GRACEFUL, linewidth=0.8, linestyle=':')
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim)
        rho = udf.rank().corr(other.rank())
        ax.set_title(f'{db} {WORKLOADS[wl]} (n={len(d):,}, <1 ms: {low.sum()})\nrank corr. {rho:.2f}, '
                     f'median UDF share {d.udf_share.median():.2f}', loc='left', fontsize=10.5)
        rows.append({'figure': '6' + x[0], 'db': db, 'workload': wl, 'n': len(d), 'spearman': rho,
                     'median_udf_share': d.udf_share.median(), 'n_udf_under_1ms': int(low.sum())})
    for ax in axes[1]:
        ax.set_xlabel('rest of the query (s)' if x == 'rest' else 'total query time (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('UDF time (s)')
    figlegend(fig, *axes[0, 0].get_legend_handles_labels(), loc='upper right')
    share = pd.concat([d.udf_share for d in q.values()])
    if x == 'rest':
        head = 'UDF time vs the rest of the query, one dot per query'
        note = 'Above the dotted diagonal: the UDF takes longer than the rest of the query. '
        name = '6a_udf_vs_rest_of_query'
    else:
        head = f'UDF time vs total query time: the UDF is most of the query (median share {share.median():.2f})'
        note = 'Dotted diagonal: UDF time = total time (a dot cannot be above it). '
        name = '6b_udf_vs_total_time'
    title(fig, head, note + 'Orange crosses: UDF under 1 ms (figure 8). Run 2 (1 thread, real seconds). ' + SAMPLE)
    fig.tight_layout()
    save(fig, out, name)


def scatter_panel(ax, d, show_legend):
    udf = d.udf_time_s.clip(lower=1e-5)
    ax.axvspan(1e-5, 1e-3, color=GRID, alpha=0.6, zorder=0, linewidth=0)
    ax.scatter(udf, d.q_error_base, s=6, color=BASE, alpha=0.45, linewidths=0, label='Baseline (one query)')
    ax.scatter(udf, d.q_error_ours, s=6, color=OURS, alpha=0.35, linewidths=0, label='Augmented (one query)')
    g = binned(d, 'udf_time_s')
    g['x'] = g.x.clip(lower=1e-5)
    big, small = g[g.n >= MIN_N], g[g.n < MIN_N]
    for col, color, name in [('base', '#5d6b78', 'Baseline'), ('ours', '#174f94', 'Augmented')]:
        ax.plot(big.x, big[col], color=color, marker='o', markersize=5, linewidth=1.6, label=f'{name}, bin median (q50)')
        ax.scatter(small.x, small[col], s=28, facecolors='none', edgecolors=color, linewidths=1.1, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-5, 100)
    hi = max(d.q_error_base.quantile(0.995), d.q_error_ours.quantile(0.995)) * 1.5
    ax.set_ylim(1, hi)
    # n per bin, staggered in two rows near the bottom so neighbouring narrow bins do not overlap
    for i, (b, r) in enumerate(g.iterrows()):
        k = LABELS.index(b)
        lo_, hi_ = max(BINS[k], 1e-5), min(BINS[k + 1], 100)
        cx = np.sqrt(lo_ * hi_)
        ax.text(cx, 0.015 + 0.05 * (k % 2), f'n={int(r.n)}', ha='center', va='bottom', fontsize=7.5,
                color=INK2, transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle='round,pad=0.15', facecolor=SURFACE, edgecolor='none', alpha=0.8))
    for e in BINS[1:-1]:
        ax.axvline(e, color=GRID, linewidth=0.8, zorder=0)
    n_above = int(((d.q_error_base > hi) | (d.q_error_ours > hi)).sum())
    if n_above:
        ax.text(0.98, 0.98, f'{n_above} {"query" if n_above == 1 else "queries"} above the axis',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color=INK2)
    if show_legend:
        ax.legend(loc='upper left', fontsize=8.5, markerscale=1.5)


def fig_scatter(q, out):
    fig, axes, cells = grid8((17, 8.4), sharex=True)
    for (db, wl), ax in cells:
        scatter_panel(ax, q[db, wl], show_legend=False)
        d = q[db, wl]
        ax.set_title(f'{db} {WORKLOADS[wl]} (n={len(d):,})\nq50 {d.q_error_base.median():.2f} -> '
                     f'{d.q_error_ours.median():.2f} ({gain(d.q_error_base.median(), d.q_error_ours.median()):+.1f}%)',
                     loc='left', fontsize=10.5)
    for ax in axes[1]:
        ax.set_xlabel('UDF time (s, log)')
    for ax in axes[:, 0]:
        ax.set_ylabel('q-error (log)')
    h, l = axes[0, 0].get_legend_handles_labels()
    figlegend(fig, h, l, loc='upper right', ncol=4, bbox_to_anchor=(1, 1.0), markerscale=2)
    title(fig, 'q-error of each query by its UDF time, baseline vs augmented',
          'x: time inside the UDF operator (run 2, 1 thread, real seconds); y: the whole query\'s q-error. Lines: '
          'median q-error (q50) per UDF-time bin, drawn at the bin\'s median UDF time; hollow: bins with fewer than '
          f'{MIN_N} queries, too few to read; n = queries per bin. Grey band: '
          f'UDF under 1 ms (figure 8). Title: q50 baseline -> augmented on these queries. {SAMPLE}')
    fig.tight_layout()
    save(fig, out, '7_qerror_vs_udf_time')
    for db in DBS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
        for ax, wl in zip(axes, WORKLOADS):
            d = q[db, wl]
            scatter_panel(ax, d, show_legend=wl == 'pushdown')
            ax.set_title(f'{WORKLOADS[wl]} (n={len(d):,}): q50 {d.q_error_base.median():.2f} -> '
                         f'{d.q_error_ours.median():.2f} ({gain(d.q_error_base.median(), d.q_error_ours.median()):+.1f}%)',
                         loc='left', fontsize=11)
            ax.set_xlabel('UDF time (s, log)')
        axes[0].set_ylabel('q-error (log)')
        title(fig, f'{db}: q-error of each query by its UDF time, baseline vs augmented',
              f'Lines: q50 per UDF-time bin; hollow: bins with fewer than {MIN_N} queries; n = queries per bin; grey band: '
              'UDF under 1 ms. Run 2 (1 thread). '
              + SAMPLE, note_y=-0.03)
        fig.tight_layout()
        save(fig, out, f'7_qerror_vs_udf_time_{db}')


def udf_tables(sql, udf):
    """Tables whose columns the UDF reads. Pull-up calls it on the joined subquery (nested_query.<col>): then the
    subquery's joined tables, prefixed 'nested:'."""
    m = re.search(re.escape(udf) + r'\((.*?)\)', sql)
    if not m:
        return ''
    direct = sorted(set(re.findall(r'"(\w+)"\."\w+"', m.group(1))))
    if direct:
        return ','.join(direct)
    return 'nested:' + ','.join(sorted(set(re.findall(r'(?:FROM|JOIN) "(\w+)"', sql))))


def low_runtime(q, out, rows):
    keep = ['db', 'workload', 'query_id', 'udf', 'udf_tables', 'rows_into_udf', 'udf_in_est_rows', 'udf_num_loops',
            'udf_num_branches', 'udf_pos_in_query', 'num_joins', 'udf_time_s', 'total_time_s', 'graceful_runtime_s',
            'q_error_base', 'q_error_ours']
    lows = []
    for (db, wl), d in q.items():
        d = d[d.udf_time_s < 0.001].copy()
        d['db'], d['workload'] = db, wl
        d['udf_tables'] = [udf_tables(s, u) for s, u in zip(d.sql, d.udf)]
        lows.append(d[keep])
    low = pd.concat(lows)
    low.to_csv(f'{out}/low_runtime_udfs.csv', index=False)

    fig, axes, cells = grid8((16, 7.6), sharex=True, sharey=True)
    lim = [1, 1e9]
    for (db, wl), ax in cells:
        d = q[db, wl]
        lo = d.udf_time_s < 0.001
        act = d.rows_into_udf.clip(lower=1)
        est = d.udf_in_est_rows.clip(lower=1)
        ax.scatter(act[~lo], est[~lo], s=5, color=BASE, alpha=0.4, linewidths=0, label='UDF >= 1 ms')
        ax.scatter(act[lo], est[lo], s=16, color=GRACEFUL, marker='x', linewidths=0.9, label='UDF < 1 ms')
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim)
        t = f'{db} {WORKLOADS[wl]}: <1 ms {lo.sum()} of {len(d):,}'
        if lo.sum():
            t += f'\nmedian rows actual {act[lo].median():,.0f}, est. {est[lo].median():,.0f}'
            rows.append({'figure': '8', 'db': db, 'workload': wl, 'n': int(lo.sum()),
                         'median_rows_actual': act[lo].median(), 'median_rows_est': est[lo].median(),
                         'q50_base': d.q_error_base[lo].median(), 'q50_ours': d.q_error_ours[lo].median(),
                         'tables': ';'.join(sorted(set(','.join(low[(low.db == db) & (low.workload == wl)].udf_tables).split(','))))})
        else:
            t += '\n'
        ax.set_title(t, loc='left', fontsize=10.5)
    for ax in axes[1]:
        ax.set_xlabel('actual rows into the UDF')
    for ax in axes[:, 0]:
        ax.set_ylabel("DuckDB's estimate")
    figlegend(fig, *axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=2, markerscale=2)
    title(fig, "UDFs under 1 ms: how many rows actually reach the UDF vs DuckDB's estimate (what the model sees)",
          "Estimate: DuckDB's cardinality estimate of the operator feeding the UDF, from the executed plan. Dotted: "
          'estimate = actual. The full list (tables, loops, both q-errors) is in low_runtime_udfs.csv. Run 2. ' + SAMPLE)
    fig.tight_layout()
    save(fig, out, '8_low_runtime_udfs')


def fig_distributions(q, out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    series = [('udf_time_s', 'UDF time (lab PC)', OURS), ('total_time_s', 'total query time (lab PC)', '#174f94'),
              ('graceful_runtime_s', "GRACEFUL's label", GRACEFUL)]
    for ax, wl in zip(axes, WORKLOADS):
        pos, labels = [], []
        for i, db in enumerate(DBS):
            d = q[db, wl]
            for j, (col, name, color) in enumerate(series):
                v = np.log10(d[col].clip(lower=1e-5))
                p = i * 4 + j
                b = ax.boxplot(v, positions=[p], widths=0.75, patch_artist=True, showfliers=True,
                               flierprops=dict(marker='o', markersize=2.5, markerfacecolor=color, markeredgecolor='none', alpha=0.6),
                               medianprops=dict(color=INK, linewidth=1.3), whiskerprops=dict(color=INK2),
                               capprops=dict(color=INK2), boxprops=dict(facecolor=color, edgecolor=INK2, alpha=0.85))
                rows.append({'figure': '9', 'db': db, 'workload': wl, 'series': col, 'n': len(d),
                             'median_s': d[col].median(), 'p25_s': d[col].quantile(.25), 'p75_s': d[col].quantile(.75)})
            pos.append(i * 4 + 1)
            labels.append(f'{db}\nn={len(q[db, wl]):,}')
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_title(WORKLOADS[wl], loc='left')
        ax.grid(axis='x', visible=False)
        ax.axhline(-3, color=GRACEFUL, linewidth=0.8, linestyle=':')
    ticks = [-5, -4, -3, -2, -1, 0, 1, 2]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(['10 µs', '0.1 ms', '1 ms', '10 ms', '0.1 s', '1 s', '10 s', '100 s'])
    axes[0].set_ylabel('time per query (log)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85) for _, _, c in series]
    figlegend(fig, handles, [n for _, n, _ in series], loc='upper right', ncol=3)
    title(fig, 'Runtime distributions across the 4 databases',
          'Boxes: 25-75% of the queries, line: median, whiskers: 1.5 x the box, dots: outliers beyond the whiskers. Lab PC: run 2 '
          "(1 thread, real seconds). GRACEFUL's label is from their machine (2 x 10 cores, 20 threads) and is about 3x "
          'our time (figure 3); every label in these sets lies between 0.1 and 30 s. Dotted: 1 ms. ' + SAMPLE)
    fig.tight_layout()
    save(fig, out, '9_runtime_distributions')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile_dir', default='results/udf_query_profile_labpc')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--ours_predictions', default='',
                   help='a folder of newer augmented predictions that replace the sheet\'s (per database, where present); '
                        'default "": the full-data sheets already hold the best checkpoints\' predictions. The sample '
                        'figures (2026-11-16) used results/udf_query_profile_labpc/newer_checkpoints/predictions')
    p.add_argument('--bare', action='store_true', help='no figure title or note (for slides)')
    args = p.parse_args()
    global BARE, DROP_SMALL
    BARE = DROP_SMALL = args.bare
    if BARE:      # slide versions: draw at 80% size, so the same font sizes read ~25% larger on the slide
        subplots = plt.subplots

        def smaller(*a, figsize=None, **k):
            return subplots(*a, figsize=None if figsize is None else (figsize[0] * 0.8, figsize[1] * 0.8), **k)
        plt.subplots = smaller
    out = args.out_dir or f'{args.profile_dir}/figures'
    os.makedirs(out, exist_ok=True)
    style()
    q1 = load(args.profile_dir, RUN1, args.ours_predictions or None)
    q2 = load(args.profile_dir, RUN2, args.ours_predictions or None)
    print('augmented predictions replaced for:', sorted(OVERRIDDEN) or 'none')
    # q-errors do not depend on the run: the same predictions are merged into both
    for k in q1:
        m = q1[k][['query_id', 'q_error_base', 'q_error_ours']].merge(q2[k][['query_id', 'q_error_base', 'q_error_ours']],
                                                                       on='query_id')
        assert len(m) == len(q1[k]) and np.allclose(m.q_error_base_x, m.q_error_base_y), k
    rows = []
    for k, d in q2.items():
        rows.append({'figure': 'all', 'db': k[0], 'workload': k[1], 'n': len(d), 'q50_base': d.q_error_base.median(),
                     'q50_ours': d.q_error_ours.median(), 'gain_pct': gain(d.q_error_base.median(), d.q_error_ours.median()),
                     'median_cv': d.total_time_cv.median(), 'median_udf_share': d.udf_share.median()})
    fig_passes(q2, out)
    fig_clock(args.profile_dir, load_runs(args.profile_dir, RUN1), load_runs(args.profile_dir, RUN2), out)
    fig_runtime(q1, out, rows)
    fig_binned(q2, out, 'total_time_s', '4a_q50_by_query_time', 'q50 by total query time, baseline vs augmented',
               'Bins: total time of the query.', rows)
    fig_binned(q2, out, 'udf_time_s', '4b_q50_by_udf_time', 'q50 by the time spent inside the UDF, baseline vs augmented',
               'Bins: time of the UDF operator; the q-error is of the whole query.', rows)
    fig_gain_trend(q2, out, rows)
    fig_udf_type(q2, out, rows)
    fig_udf_sql(q2, out, 'rest', rows)
    fig_udf_sql(q2, out, 'total', rows)
    fig_scatter(q2, out)
    low_runtime(q2, out, rows)
    fig_distributions(q2, out, rows)
    pd.DataFrame(rows).to_csv(f'{out}/summary.csv', index=False)
    print('figures written to', out)


if __name__ == '__main__':
    main()
