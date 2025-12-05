from lightautoml.dataset.roles import (
    NumericRole,
    CategoryRole,
    DatetimeRole,
    GroupRole,
    TargetRole
)

# Numeric transformers
from lightautoml.transformers.numeric import (
    FillnaMedian, 
    FillnaMean,
    FillInf,
    StandardScaler,
)

# Base transformers
from lightautoml.transformers.base import (
    SequentialTransformer,
    ColumnsSelector,
)

from lightautoml.transformers.decomposition import (
    PCATransformer
)

from lightautoml.automl.base import AutoML
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.report import ReportDeco
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.tasks import Task

import pandas as pd

RANDOM_STATE = 42
N_FOLDS = 5

roles = {
    'target': 'Sales',
    'group': 'id_2',
    NumericRole(): [
        'Promo', 'SchoolHoliday', 'CompetitionDistance',
        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
        'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'
    ],
    CategoryRole(): [
        'Store', 'StateHoliday', 'StoreType', 
        'Assortment', 'PromoInterval'
    ],
    DatetimeRole(base_date=True, seasonality=['year', 'month', 'week', 'dayofweek']): 'Date'
}

task = Task('reg')
train = pd.read_csv('example_train.csv').sample(frac=0.01)
reader = PandasToPandasReader(
    task, 
    cv=N_FOLDS, 
    random_state=RANDOM_STATE,
    advanced_roles=True
)

tr1 = SequentialTransformer([
    SequentialTransformer([
        ColumnsSelector(['CompetitionDistance', 'Promo', 'SchoolHoliday', 'Promo2']),
        FillInf(),
        FillnaMedian(),
        StandardScaler()
    ])
])

tr2 = SequentialTransformer([
    SequentialTransformer([
        ColumnsSelector(['CompetitionDistance', 'Promo', 'SchoolHoliday', 'Promo2']),
        FillnaMean(),
        StandardScaler(),
        #PCATransformer(n_components=5)
    ])
])

lgbm_tuner = OptunaTuner(n_trials=20, timeout=30)
pipeline_lgbm = MLPipeline(
    [(BoostLGBM(), lgbm_tuner)],
    features_pipeline=tr1
)

cb_tuner = OptunaTuner(n_trials=20, timeout=30)
pipeline_cb = MLPipeline(
    [(BoostCB(), cb_tuner)],
    features_pipeline=tr2
)

cb2_tuner = OptunaTuner(n_trials=20, timeout=30)
pipeline_cb2 = MLPipeline(
    [(BoostCB(), cb2_tuner)],
    features_pipeline=LGBSimpleFeatures()
)

automl = AutoML(
    reader,
    [
        [pipeline_lgbm, pipeline_cb],  # уровень 1
        [pipeline_cb2]
    ]
)

#rd = ReportDeco(output_path='reports/custom_report.html')
#automl_reported = rd(automl)

oof_pred = automl.fit_predict(
    train,
    roles=roles,
    verbose=4
)