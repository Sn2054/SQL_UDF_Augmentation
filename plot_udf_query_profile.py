"""Figures for the per-query profile (profile_udf_queries.py output). CPU only, reads the spreadsheets and run CSVs.

    CUDA_VISIBLE_DEVICES="" python plot_udf_query_profile.py \
        --profile_dir results/udf_query_profile_clean --out_dir results/udf_query_profile_clean/figures

Figures (PNG + PDF):
  1_measurement_passes   how much a query's time varies across the 3 passes (coefficient of variation)
  2_machine_state        clock speed and core temperature over the whole run (no throttling check)
  3_runtime_vs_graceful  mean time per query, GRACEFUL's recorded runtime vs ours
  4a_q50_by_query_time   q50 of the baseline vs ours by total query runtime, per DB and workload
  4b_q50_by_udf_time     the same by the time spent in the UDF
  5_q50_by_udf_type      q50 of the baseline vs ours by where the UDF sits (WHERE / SELECT) and its control flow
  6a_udf_vs_rest_of_query  UDF time vs the rest of the query's time, one dot per query
  6b_udf_vs_total_time     UDF time vs total query time
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

DBS = ['employee', 'financial']
WORKLOADS = {'pushdown': 'push-down', 'pullup': 'pull-up'}
# reference palette: slot 1 for ours, a neutral grey for the baseline reference
OURS, BASE, GRACEFUL = '#2a78d6', '#98a6b3', '#eb6834'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
TIME_BINS = [0, 0.1, 0.5, 2, 10, np.inf]
TIME_LABELS = ['<0.1 s', '0.1-0.5 s', '0.5-2 s', '2-10 s', '>10 s']
MIN_N = 20   # buckets with fewer queries are left out

FONT = '/mnt/data/shehryar/tools/libreoffice/root/opt/libreoffice26.2/share/fonts/truetype/LiberationSerif-Regular.ttf'


def style():
    if os.path.exists(FONT):
        for f in ['Regular', 'Bold', 'Italic']:
            font_manager.fontManager.addfont(FONT.replace('Regular', f))
        family = 'Liberation Serif'   # metric-compatible with Times New Roman (the deck font)
    else:
        family = 'serif'
    plt.rcParams.update({
        'font.family': family, 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.axisbelow': True, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'legend.frameon': False,
    })


def load(profile_dir):
    """{(db, workload): per-query frame with both models' q-errors}, {(db, workload): per-run frame}."""
    q, runs = {}, {}
    for db in DBS:
        for wl in WORKLOADS:
            b = pd.read_excel(f'{profile_dir}/{db}/{db}_baseline.xlsx', sheet_name=wl)
            a = pd.read_excel(f'{profile_dir}/{db}/{db}_augmented.xlsx', sheet_name=wl)[['query_id', 'q_error']]
            q[db, wl] = b.merge(a, on='query_id', suffixes=('_base', '_ours'))
            r = pd.read_csv(f'{profile_dir}/{db}/{db}_{wl}_runs.csv', parse_dates=['started'])
            runs[db, wl] = r
    return q, runs


def save(fig, out_dir, name):
    for ext in ['png', 'pdf']:
        fig.savefig(f'{out_dir}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def bars(ax, labels, base, ours, ns=None, gain=True):
    """Grouped bars baseline vs ours with the % change printed above each pair."""
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, base, w, color=BASE, label='Baseline')
    ax.bar(x + w / 2 + 0.01, ours, w, color=OURS, label='Ours (augmented)')
    top = max(max(base), max(ours))
    for i in range(len(labels)):
        if gain:
            g = round(100 * (base[i] - ours[i]) / base[i]) + 0   # + 0 turns -0 into 0
            ax.text(x[i], max(base[i], ours[i]) + top * 0.03, f'{g:+.0f}%', ha='center', va='bottom',
                    fontsize=10, color=INK, fontweight='bold' if g > 0 else 'normal')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(labels, ns)] if ns is not None else labels, fontsize=9.5)
    ax.set_ylim(1, top * 1.18)   # q-error starts at 1
    ax.grid(axis='x', visible=False)


def fig_passes(q, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (key, d) in zip(axes, q.items()):
        cv = d.total_time_cv.clip(lower=1e-4) * 100
        ax.hist(cv, bins=np.logspace(-2, 2, 41), color=OURS, edgecolor=SURFACE, linewidth=0.6,
                weights=np.full(len(cv), 100 / len(cv)))
        ax.set_xscale('log')
        ax.set_xticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])
        ax.axvline(cv.median(), color=INK, linewidth=1.2, linestyle='--')
        ax.set_title(f'{key[0]} {WORKLOADS[key[1]]}  (n={len(cv):,})')
        ax.text(0.97, 0.95, f'median {cv.median():.1f}%', transform=ax.transAxes, ha='right', va='top',
                fontsize=10, color=INK)
        ax.set_xlabel("difference between a query's 3 runs (%)")
    axes[0].set_ylabel('% of queries')
    fig.suptitle('Each query was run 3 times (in order, reversed, random); its 3 times barely differ',
                 x=0.01, ha='left', fontsize=13, color='#26326c')
    fig.text(0.01, -0.03, "x: spread of a query's 3 times (standard deviation) as a % of their mean; e.g. 1.00, 1.01 "
             "and 0.99 s give about 1%. Each bar: the % of the queries whose runs differ by that much.",
             fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, out_dir, '1_measurement_passes')


def fig_machine(runs, out_dir):
    r = pd.concat(runs.values()).sort_values('started')
    t0 = r.started.min()
    hours = (r.started - t0).dt.total_seconds() / 3600
    gap = hours.diff() > 5 / 60          # no line across pauses longer than 5 min (model loading between parts)
    freq = r.cpu_freq_mean_mhz.where(~gap)
    temp = r.core_temp_max_c.where(~gap)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    a1.plot(hours, freq / 1000, color=OURS, linewidth=0.6)
    a1.set_ylabel('clock (GHz)')
    a1.set_ylim(0, 4.2)
    a1.axhline(2.9, color=INK2, linewidth=1, linestyle=':')
    a1.text(hours.min() + 0.05, 2.75, 'base clock 2.9 GHz', ha='left', va='top', fontsize=9.5, color=INK2)
    a2.plot(hours, temp, color=GRACEFUL, linewidth=0.6)
    a2.set_ylabel('core temp. (°C)')
    a2.set_ylim(30, 100)
    a2.set_xlabel('hours since start of the run')
    # where each part of the run begins
    for i, ((db, wl), d) in enumerate(sorted(runs.items(), key=lambda kv: kv[1].started.min())):
        h = (d.started.min() - t0).total_seconds() / 3600
        for ax in (a1, a2):
            ax.axvline(h, color=GRID, linewidth=1.5, zorder=0)
        a2.text(h + 0.08, 32 if i % 2 == 0 else 44, f'{db} {WORKLOADS[wl]}', fontsize=8.5, color=INK2, va='bottom')
    fig.suptitle(f'The measuring core ran at full speed and stayed cool: median {r.cpu_freq_mean_mhz.median()/1000:.2f} GHz, '
                 f'max {r.core_temp_max_c.max():.0f} °C ({len(r):,} runs)', x=0.01, ha='left', fontsize=13, color='#26326c')
    fig.tight_layout()
    save(fig, out_dir, '2_machine_state')


def fig_runtime(q, out_dir):
    labels = [f'{db}\n{WORKLOADS[wl]}' for db, wl in q]
    g = [d.graceful_runtime_s.mean() for d in q.values()]
    o = [d.total_time_s.mean() for d in q.values()]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2 - 0.01, g, w, color=GRACEFUL, label='GRACEFUL (recorded runtime)')
    ax.bar(x + w / 2 + 0.01, o, w, color=OURS, label='Ours (this server, mean of 3 passes)')
    for i in range(len(x)):
        ax.text(x[i] - w / 2, g[i] + 0.15, f'{g[i]:.2f} s', ha='center', fontsize=10, color=INK)
        ax.text(x[i] + w / 2, o[i] + 0.15, f'{o[i]:.2f} s', ha='center', fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('mean time per query (s)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right')
    ax.set_title('Mean runtime per query, same queries', loc='left', color='#26326c', fontsize=13)
    fig.tight_layout()
    save(fig, out_dir, '3_runtime_vs_graceful')


def fig_length(q, out_dir, by):
    """by='query': buckets by total query time (4a); by='udf': by the time inside the UDF (4b)."""
    col = 'total_time_s' if by == 'query' else 'udf_time_s'
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for ax, ((db, wl), d) in zip(axes.flat, q.items()):
        d = d.assign(bucket=pd.cut(d[col], TIME_BINS, labels=TIME_LABELS, include_lowest=True))
        g = d.groupby('bucket', observed=True).agg(n=('q_error_base', 'size'), base=('q_error_base', 'median'),
                                                    ours=('q_error_ours', 'median'))
        g = g[g.n >= MIN_N]
        bars(ax, list(g.index), g.base.values, g.ours.values, g.n.values)
        ax.set_title(f'{db} {WORKLOADS[wl]}', loc='left')
        ax.set_ylabel('q50 (median q-error)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=2, bbox_to_anchor=(1, 1.0))
    if by == 'query':
        title = 'q50 by query runtime (whole query, this server), % = change vs the baseline'
        note = 'Buckets: total time of the query (1 thread).'
        name = '4a_q50_by_query_time'
    else:
        title = 'q50 by UDF runtime (time inside the UDF, this server), % = change vs the baseline'
        note = 'Buckets: time spent in the UDF operator per query (1 thread). The q-error is of the whole query.'
        name = '4b_q50_by_udf_time'
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, color='#26326c')
    fig.text(0.01, -0.01, f'{note} Buckets with fewer than {MIN_N} queries are left out.', fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, out_dir, name)


def fig_udf_type(q, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for ax, ((db, wl), d) in zip(axes.flat, q.items()):
        pos = d.udf_pos_in_query.map(lambda p: 'SELECT' if p == 'select' else 'WHERE')
        ctrl = np.where(d.udf_num_loops > 0, 'loop', np.where(d.udf_num_branches > 0, 'branch', 'plain'))
        d = d.assign(group=pos + '\n' + ctrl)
        order = [f'{p}\n{c}' for p in ['WHERE', 'SELECT'] for c in ['plain', 'loop', 'branch']]
        g = d.groupby('group').agg(n=('q_error_base', 'size'), base=('q_error_base', 'median'),
                                   ours=('q_error_ours', 'median')).reindex(order).dropna()
        g = g[g.n >= MIN_N]
        bars(ax, list(g.index), g.base.values, g.ours.values, g.n.astype(int).values)
        ax.set_title(f'{db} {WORKLOADS[wl]}', loc='left')
        ax.set_ylabel('q50 (median q-error)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', ncol=2, bbox_to_anchor=(1, 1.0))
    fig.suptitle('q50 by where the UDF is and its control flow, % = change vs the baseline', x=0.01, ha='left',
                 fontsize=13, color='#26326c')
    fig.text(0.01, -0.01, f'WHERE / SELECT: where the UDF is in the query; plain: UDF without loop or branch. Groups with '
             f'fewer than {MIN_N} queries are left out (pull-up has the UDF in WHERE only).',
             fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, out_dir, '5_q50_by_udf_type')


def fig_udf_sql(q, out_dir, x):
    """x='rest': UDF time vs the rest of the query (6a); x='total': UDF time vs total query time (6b)."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    for ax, ((db, wl), d) in zip(axes, q.items()):
        sql = (d.sql_time_s if x == 'rest' else d.total_time_s).clip(lower=1e-4)
        udf = d.udf_time_s.clip(lower=1e-4)
        ax.scatter(sql, udf, s=6, color=OURS, alpha=0.35, linewidths=0)
        lim = [1e-4, 60]
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim)
        rho = pd.Series(udf).rank().corr(pd.Series(sql).rank())
        ax.set_title(f'{db} {WORKLOADS[wl]}\nrank correlation {rho:.2f}', loc='left', fontsize=11)
        ax.set_xlabel('rest of the query (s)' if x == 'rest' else 'total query time (s)')
    axes[0].set_ylabel('UDF time (s)')
    cluster = ('The low cluster (UDF < 1 ms) is a UDF filter on a small table (median 27 rows into the UDF), where the '
               'joins take the time.')
    if x == 'rest':
        title = 'UDF time vs the rest of the query, one dot per query: the UDF dominates and the two barely move together'
        note = 'Dots above the dotted line: the UDF takes longer than the rest of the query. ' + cluster
        name = '6a_udf_vs_rest_of_query'
    else:
        title = ('UDF time vs total query time, one dot per query: most dots sit on the line, i.e. the UDF is almost '
                 'the whole query')
        note = 'Dotted line: UDF time = total time (a dot cannot be above it). ' + cluster
        name = '6b_udf_vs_total_time'
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, color='#26326c')
    fig.text(0.01, -0.02, note, fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, out_dir, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile_dir', default='results/udf_query_profile_clean')
    p.add_argument('--out_dir', default=None)
    args = p.parse_args()
    out = args.out_dir or f'{args.profile_dir}/figures'
    os.makedirs(out, exist_ok=True)
    style()
    q, runs = load(args.profile_dir)
    fig_passes(q, out)
    fig_machine(runs, out)
    fig_runtime(q, out)
    fig_length(q, out, 'query')
    fig_length(q, out, 'udf')
    fig_udf_type(q, out)
    fig_udf_sql(q, out, 'rest')
    fig_udf_sql(q, out, 'total')
    print('figures written to', out)


if __name__ == '__main__':
    main()
