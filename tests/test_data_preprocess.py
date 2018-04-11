import pandas as pd
from numpy import nan


def test_clean_nan():
    from qrnn.data_preprocess import clean_nan
    df = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-01", "2017-01-02", nan, "2017-01-04"]),
        'price1': [100.0, 102.0, 94.0, nan],
        "price2": [1010.0, 1022.0, 500.0, 501.0]
    })

    expected = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-01", "2017-01-02"]),
        'price1': [100.0, 102.0],
        "price2": [1010.0, 1022.0]
    })

    result = clean_nan(df)
    assert result.equals(expected), "clean_nan not working"

def test_lagger():
    from qrnn.data_preprocess import lagger
    df = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"]),
        'price1': [100.0, 102.0, 94.0, 105.0],
        "price2": [1010.0, 1022.0, 500.0, 501.0]
    })

    expected = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"]),
        'price1':  [100.0,  102.0,  94.0,  105.0],
        'price11': [102.0,  94.0,   105,   nan],
        'price12': [94.0,   105.0,  nan,   nan],
        "price2":  [1010.0, 1022.0, 500.0, 501.0],
        "price21": [1022.0, 500.0,  501.0, nan],
        "price22": [500.0,  501.0,  nan, nan]
    })

    lagg_fn = lagger(n_lags=2, price_columns=['price1', 'price2'])
    result = lagg_fn(df)
    print(result)
    assert result.equals(expected), "lagger not working"

def test_diff_log_pricer():
    from qrnn.data_preprocess import diff_log_pricer
    from numpy import log

    df_sorted = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"]),
        'price1': [100, 102, 94, 105],
        "price2": [1010, 1022, 500, 501]
    })

    df_copy = df_sorted.copy()

    df_not_sorted = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-04", "2017-01-01", "2017-01-02", "2017-01-03"]),
        'price1': [105, 100, 102, 94],
        "price2": [501, 1010, 1022, 500]
    })

    expected = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"]),
        'price1': [None, 100 * (log(102) - log(100)), 100 * (log(94) - log(102)), 100 * (log(105) - log(94))],
        "price2": [None, 100 * (log(1022) - log(1010)), 100 * (log(500) - log(1022)), 100 * (log(501) - log(500))]
    })

    difflog_fn = diff_log_pricer(price_columns=["price1", "price2"],
                                 date_column="date")

    result_sorted = difflog_fn(df_sorted)
    result_not_sorted = difflog_fn(df_not_sorted)

    assert result_sorted.equals(expected), "diff_log_pricer not working on sorted df"
    assert result_not_sorted.equals(expected), "diff_log_pricer not working on not sorted df"
    assert df_copy.equals(df_sorted), "input got changed during transforms"


def test_time_split_dataset():
    from qrnn.data_preprocess import time_split_dataset

    df = pd.DataFrame({
        'col': [1, 2, 3, None],
        'date': pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"])
    })

    spliter_fn = time_split_dataset(train_start_date="2017-01-01",
                                    train_end_date="2017-01-02",
                                    holdout_end_date="2017-01-04",
                                    date_col="date")

    in_time_train_set, out_time_test_set = spliter_fn(df)

    expected_train = pd.DataFrame({
        'col': [1., 2],
        'date': pd.to_datetime(["2017-01-01", "2017-01-02"])
    })

    expected_test = pd.DataFrame({
        'col': [3., None],
        'date': pd.to_datetime(["2017-01-03", "2017-01-04"])
    })

    assert in_time_train_set.reset_index(drop=True).equals(expected_train)
    assert out_time_test_set.reset_index(drop=True).equals(expected_test)
