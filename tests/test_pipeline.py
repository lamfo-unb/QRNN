import pandas as pd
from pandas.testing import assert_frame_equal
from qrnn.pipeline import pipeline
from qrnn.data_preprocess import *


def test_pipeline():
    df = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"]),
        'price1': [100.0, 102.0, 94.0, 80.0],
        'price2': [1010.0, 1022.0, 500.0, 501.0]
    })

    expected = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-02", "2017-01-03"]),
        'price1' : [1.980263, -8.167803],
        'price11': [-8.167803, -16.126815],
        'price2' : [1.181116, -71.490867],
        'price21': [-71.490867, 0.199800]
    })

    PRICE_COLS = ["price1", "price2"]
    differ_learner = diff_log_pricer(price_columns=PRICE_COLS, date_column="date")
    lagger_learner = lagger(n_lags=1, price_columns=PRICE_COLS)
    na_clearn_learner = clean_nan(how="any")

    pipeline_fn = pipeline(learners=[differ_learner,
                                     lagger_learner,
                                     na_clearn_learner])
    result = pipeline_fn(df)
    result.index = expected.index

    assert_frame_equal(result, expected)
