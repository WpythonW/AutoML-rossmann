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

from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

from lightautoml.transformers.base import LAMLTransformer
from lightautoml.dataset.base import LAMLDataset
import numpy as np

class NonZeroFilter(LAMLTransformer):
    """Фильтрует строки с нулевыми значениями только при обучении"""
    
    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = None  # не меняем имена фичей
    
    def __init__(self, feature_col='A'):
        self.feature_col = feature_col
    
    def fit(self, dataset: LAMLDataset) -> 'NonZeroFilter':
        """Запоминаем индексы для фильтрации"""
        self.features = dataset.features
        
        # Получаем значения колонки
        data = dataset.to_pandas().data
        feature_vals = data[self.feature_col].values
        
        # Маска для A > 0
        self.train_mask = feature_vals > 0
        
        return self
    
    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """На transform ничего не фильтруем (для predict)"""
        return dataset
    
    def fit_transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """При fit_transform фильтруем A=0"""
        self.features = dataset.features
        
        # Конвертируем в pandas
        dataset = dataset.to_pandas()
        data = dataset.data
        
        # Фильтрация
        feature_vals = data[self.feature_col].values
        mask = feature_vals > 0
        
        filtered_data = data[mask].reset_index(drop=True)
        
        # Фильтруем target и другие атрибуты
        filtered_attrs = {}
        for attr in dataset._array_like_attrs:
            if hasattr(dataset, attr) and getattr(dataset, attr) is not None:
                attr_vals = getattr(dataset, attr)
                filtered_attrs[attr] = attr_vals[mask]
        
        # Создаем новый dataset
        output = dataset.empty()
        output.set_data(filtered_data, dataset.features, dataset.roles)
        
        # Восстанавливаем атрибуты
        for attr, val in filtered_attrs.items():
            setattr(output, attr, val)
        
        return output


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
    advanced_roles=False,  # явно False
    n_jobs=1  # отключаем параллелизм для угадывания
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
        #[pipeline_cb2]
    ]
)

#rd = ReportDeco(output_path='reports/custom_report.html')
#automl_reported = rd(automl)

oof_pred = automl.fit_predict(
    train,
    roles=roles,
    verbose=4
)