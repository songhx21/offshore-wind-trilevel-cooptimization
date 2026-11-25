"""Recompute paired statistics and supplementary plots from archived records."""
from pathlib import Path
import argparse
import csv
import itertools
import json
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def holm(values):
    order = np.argsort(values)
    result = np.empty(len(values))
    previous = 0.0
    for position, i in enumerate(order):
        previous = max(previous, min(1.0, (len(values)-position)*values[i]))
        result[i] = previous
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis_output')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / 'data/run_metrics.csv').open(encoding='utf-8')))
    index = {(int(r['turbines']), int(r['version']), r['workflow'], int(r['seed'])): r for r in rows}
    result = []
    for size, seeds in [(25, [10,42,98,133,202,314,777,1024]), (50, [42,98,133,202])]:
        for baseline in ['BASE', 'SEQ', 'CO']:
            family = []
            for version in [7,8,9,10]:
                if size == 50 and baseline == 'BASE' and version > 7:
                    continue
                left = [index[size,version,baseline,s] for s in seeds]
                right = [index[size,version,'TCO',s] for s in seeds]
                a = np.array([float(r['score_million_units']) for r in left])
                b = np.array([float(r['score_million_units']) for r in right])
                gain = 100*(a-b)/np.abs(a)
                rng = np.random.default_rng(20261029)
                samples = gain[rng.integers(0,len(seeds),(20000,len(seeds)))].mean(axis=1)
                strict = np.array([x['strict_pass'] == y['strict_pass'] == 'True' for x,y in zip(left,right)])
                family.append({'turbines':size, 'configuration':{7:'AWS',8:'FAE',9:'RUC',10:'RNT'}[version],
                               'baseline':baseline, 'paired_n':len(seeds), 'score_reduction_pct':float(gain.mean()),
                               'bootstrap_low_pct':float(np.quantile(samples,.025)), 'bootstrap_high_pct':float(np.quantile(samples,.975)),
                               'wins':int((gain > 0).sum()), 'strict_pair_n':int(strict.sum()),
                               'strict_pair_score_reduction_pct':float(gain[strict].mean()),
                               'p_exact':float(wilcoxon(a-b,method='exact',alternative='two-sided').pvalue)})
            for item, adjusted in zip(family, holm([r['p_exact'] for r in family])):
                item['p_holm'] = float(adjusted)
            result.extend(family)
    with (args.output / 'paired_statistics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)
    print(json.dumps({'selected_runs':len(rows), 'paired_comparisons':len(result), 'output':str(args.output)}, indent=2))
    if not args.figures:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'BASE':'#666666','SEQ':'#0072B2','CO':'#D55E00','TCO':'#009E73'}
    plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    for size in [25,50]:
        archive = json.loads((ROOT / f'data/diagnostics_{size}t.json').read_text(encoding='utf-8'))
        fig, axes = plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
        for ax, version in zip(axes.flat,[7,8,9,10]):
            for method in colors:
                curves = []
                for key, data in archive.items():
                    if not key.startswith(f'{size}T_V{version}_{method}_'):
                        continue
                    curve = data.get('fig6_ppo_learning_curves',{}).get('episodes',[])
                    if not curve:
                        continue
                    x = [float(r['episode']) for r in curve]
                    y = [float(r['cost_m']) for r in curve]
                    grid = np.arange(0,2451,50)
                    curves.append(np.interp(grid,x,y))
                if curves:
                    ax.plot(grid,np.mean(curves,axis=0),label=method,color=colors[method],lw=1.25)
            ax.set_title({7:'AWS',8:'FAE',9:'RUC',10:'RNT'}[version])
            ax.set_xlabel('Policy-search episode')
            ax.set_ylabel('Episode score (million units)')
            ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
        for ext in ['png','pdf']:
            fig.savefig(args.output/f'learning_curves_{size}t.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
