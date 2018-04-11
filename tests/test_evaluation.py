import pandas as pd


def test_quantile_loss_evaluator():
    from qrnn.evaluation import quantile_loss_evaluator

    df = pd.DataFrame({
        'y': [4.094501, -8.509012, 5.50190, -1.51501],
        'prediction': [-2.09450, -3.509012, -5.50190, -1.51501]
    })

    eval_fn = quantile_loss_evaluator(predict_col="prediction", target_col="y", tau=0.05)

    loss = eval_fn(df)
    assert loss == 1.4024100124999999

def test_proportion_of_hits_evaluator():
    from qrnn.evaluation import proportion_of_hits_evaluator

    df = pd.DataFrame({
        'y': [4.094501, -8.509012, 5.50190, -1.51501],
        'prediction': [-2.09450, -3.509012, -5.50190, -1.51501] # one hit in the second line
    })

    eval_fn = proportion_of_hits_evaluator(predict_col="prediction", target_col="y")
    loss = eval_fn(df)

    assert loss == 1.0 / 4.0
