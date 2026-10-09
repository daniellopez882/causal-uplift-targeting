from __future__ import annotations

import numpy as np

import causal_uplift as cu


def test_package_docstring_example_runs() -> None:
    data = cu.make_synthetic("step", 4_000, seed=0)
    cate = cu.DRLearner(cu.BaseLearner(params={"n_estimators": 50})).fit(data.X, data.arm, data.y).predict(data.X)
    plan = cu.allocate(cate, budget=0.2 * data.n)
    assert cate.shape == (data.n, 2)
    assert int((plan.assign > 0).sum()) <= 0.2 * data.n
    assert set(np.unique(plan.assign)) <= {0, 1, 2}
    assert cu.__version__
