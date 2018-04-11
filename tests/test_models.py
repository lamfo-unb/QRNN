from collections import Counter

import pandas as pd


def test_qrnn_learner():
    from qrnn.models import qrnn_learner

    train = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-02", "2017-01-03", "2017-01-04", "2017-01-06"]),
        'price1': [1.980263, -8.167803, 2.16421, -2.13451],
        'price2': [2.431313, -6.512141, 1.34811, -4.21594],
        'y': [4.094501, -3.509012, 5.50190, -1.51501],
    })

    test = pd.DataFrame({
        'date': pd.to_datetime(["2017-01-07", "2017-01-08", "2017-01-09", "2017-01-10"]),
        'price1': [2.403210, -2.403153, 6.43115, -1.43201],
        'price2': [4.910245, -1.470514, 8.40329, -2.57012],
    })

    model, pred_train = qrnn_learner(dataset=train,
                                     price_cols=["price1", "price2"],
                                     target_col="y",
                                     prediction_col="prediction")

    pred_test = model(test)

    expected_col_train = train.columns.tolist() + ["prediction"]
    expected_col_test = test.columns.tolist() + ["prediction"]

    # check if model is making a prediction column
    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
