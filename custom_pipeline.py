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
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.report import ReportDeco
from lightautoml.tasks import Task

import pandas as pd

RANDOM_STATE = 42

train = pd.read_csv('example_train.csv')

roles = {
    'Store': CategoryRole(),
    'Date': DatetimeRole(base_date=True, seasonality=['year', 'month', 'week', 'dayofweek']),
    'Sales': TargetRole(),
    'Promo': NumericRole(),
    'StateHoliday': CategoryRole(),
    'SchoolHoliday': NumericRole(),
    'StoreType': CategoryRole(),
    'Assortment': CategoryRole(),
    'CompetitionDistance': NumericRole(),
    'CompetitionOpenSinceMonth': NumericRole(),
    'CompetitionOpenSinceYear': NumericRole(),
    'Promo2': NumericRole(),
    'Promo2SinceWeek': NumericRole(),
    'Promo2SinceYear': NumericRole(),
    'PromoInterval': CategoryRole(),
    'id_2': GroupRole()
}


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
        PCATransformer(n_components=5)
    ])
])

task = Task('binary')

N_FOLDS = 5
reader = PandasToPandasReader(task, cv=N_FOLDS, random_state=RANDOM_STATE)

lgbm_tuner = OptunaTuner(n_trials=20, timeout=30)
pipeline_lgbm = MLPipeline(
    [(BoostLGBM(), lgbm_tuner)],
    features_pipeline=tr1
)

pipeline_linear = MLPipeline(
    [LinearLBFGS()],
    features_pipeline=tr2
)

automl = AutoML(
    reader,
    [
        [pipeline_lgbm, pipeline_linear]  # уровень 1
    ]
)

rd = ReportDeco(output_path='reports/custom_report.html')
automl_reported = rd(automl)

oof_pred = automl_reported.fit_predict(
    train,
    roles=roles,
    verbose=4
)