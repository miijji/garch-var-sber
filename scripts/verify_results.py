"""Audit a completed full notebook run against saved observations."""
import json
from pathlib import Path
import shutil

import nbformat
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
nb = nbformat.read(root / 'outputs/garch_var_research.ipynb', as_version=4)
code = [c for c in nb.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code), 'Not all cells executed'
assert not any(o.output_type == 'error' for c in code for o in c.outputs)
out = sorted((root / 'outputs/runs').glob('notebook_*'))[-1]
pred = pd.read_csv(out / 'predictions.csv', parse_dates=['date', 'origin'])
summary = pd.read_csv(out / 'backtest_summary.csv')
actual = pd.read_csv(out / 'returns.csv', parse_dates=['date']).set_index('date').return_pct
settings = json.loads((out / 'settings.json').read_text(encoding='utf-8'))
for model, group in pred.groupby('model'):
    assert len(group) == settings['test_size']
    assert not group.date.duplicated().any()
    assert group.date.is_monotonic_increasing
    assert (group.origin < group.date).all()
    assert group.date.tolist() == actual.index[-settings['test_size']:].tolist()
    np.testing.assert_allclose(group.actual_pct, actual.loc[group.date])
    for tag in ['95', '99']:
        np.testing.assert_allclose(group[f'var_{tag}_loss_pct'], -group[f'q_{tag}_pct'])
        assert (group[f'es_{tag}_loss_pct'] >= group[f'var_{tag}_loss_pct']).all()
        k = int((group.actual_pct < group[f'q_{tag}_pct']).sum())
        row = summary[(summary.model == model) & (summary.confidence == int(tag)/100)].iloc[0]
        assert row.exceptions == k
    assert (group.q_99_pct < group.q_95_pct).all()
assert len(summary) == 6
for file in ['backtest_summary.csv', 'backtest_GARCH-normal.png', 'backtest_GARCH-t.png', 'correlations.png',
             'stationarity.csv', 'residual_diagnostics_normal.csv', 'residual_diagnostics_t.csv',
             'initial_fit_normal.txt', 'initial_fit_t.txt', 'next_observation_forecast.csv']:
    shutil.copyfile(out / file, root / 'outputs' / file)
print(summary[['model', 'confidence', 'n', 'exceptions', 'expected', 'kupiec_p', 'independence_p']].to_string(index=False))
print('Verified:', out)
print('Executed code cells:', len(code))
