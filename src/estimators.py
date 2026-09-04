"""CATE estimator roster, all sharing the same gradient-boosted base learners.

Estimators (framing's full list):
    S-learner, T-learner, X-learner   -> meta-learners  (econml.metalearners)
    DR-learner                          -> doubly-robust  (econml.dr)
    R-learner                           -> residual-on-residual DML (econml.dml.NonParamDML)
    Causal Forest (GRF)                 -> econml.dml.CausalForestDML

Every estimator uses HistGradientBoosting base models so differences in the
CATE estimates come from the *meta-structure*, not from different base learners.
Treatment is the randomized `treatment` flag (intent-to-treat); because
assignment is randomized the propensity is ~0.85 everywhere, which the
propensity models recover empirically.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

# Keep base learners modest so the full-data run stays tractable.
_HGB_KW = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
               min_samples_leaf=200, early_stopping=False)


def _reg():
    return HistGradientBoostingRegressor(**_HGB_KW)


def _clf():
    return HistGradientBoostingClassifier(**_HGB_KW)


def build_estimators(cv: int = 3, cf_estimators: int = 400, random_state: int = 0) -> dict:
    """Return {name: unfitted econml estimator}. `cv` controls cross-fitting folds."""
    from econml.metalearners import SLearner, TLearner, XLearner
    from econml.dr import DRLearner
    from econml.dml import CausalForestDML, NonParamDML

    est = {}

    # --- Meta-learners ---------------------------------------------------
    est["S-learner"] = SLearner(overall_model=_reg())
    est["T-learner"] = TLearner(models=_reg())
    est["X-learner"] = XLearner(models=_reg(), propensity_model=_clf(), cate_models=_reg())

    # --- Doubly-robust ---------------------------------------------------
    est["DR-learner"] = DRLearner(
        model_propensity=_clf(),
        model_regression=_reg(),
        model_final=_reg(),
        cv=cv,
        random_state=random_state,
    )

    # --- R-learner (NonParametric DML, residual-on-residual) -------------
    est["R-learner"] = NonParamDML(
        model_y=_reg(),
        model_t=_clf(),
        model_final=_reg(),
        discrete_treatment=True,
        cv=cv,
        random_state=random_state,
    )

    # --- Causal Forest / GRF --------------------------------------------
    # GRF requires n_estimators divisible by subforest_size (default 4).
    cf_estimators = int(np.ceil(cf_estimators / 4) * 4)
    est["Causal Forest"] = CausalForestDML(
        model_y=_reg(),
        model_t=_clf(),
        discrete_treatment=True,
        n_estimators=cf_estimators,
        min_samples_leaf=200,
        max_depth=None,
        cv=cv,
        random_state=random_state,
    )
    return est


def fit_effect(name, est, X_tr, T_tr, Y_tr, X_ev):
    """Fit one estimator and return its CATE vector on the evaluation features."""
    # econml signature: fit(Y, T, X=..., W=...). We pass X as effect modifiers;
    # for DML/DR X also serves as controls when W is None.
    est.fit(Y_tr, T_tr, X=X_tr)
    tau = est.effect(X_ev)
    return tau.ravel()
