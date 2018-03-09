import numpy as np
from toolz.curried import *

def lagg_3d(dataset, n_lags, price_columns):

    def p(new_df):
        return pipe(price_columns,
                    map(lambda col: [new_df[col].shift(-lag).tolist() for lag in range(n_lags)]),
                    compose(np.array, list),
                    partial(np.transpose, axes=(2, 1, 0)))

    return p, p(dataset)



@curry
def diff_log_pricer(dataset, price_columns, date_column):
    """
    Splits temporal data into a training and testing datasets such that
    all training data comes before the testings one.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame with a Date Column and one or many price column.
        The price column must be of numerical time and not contain nones

    price_cols : list of str
        A list with the names of the price columns

    date_col : str
        The name of the date column. The column must be of type datetime.

    Returns
    ----------
    new_df : pandas.DataFrame
        A df like DataFrame with the price column replaced by the log difference in time.
        The first row will contain NaNs due to first diferentiation.
    """

    def p(new_df):
        # Sorting the dataframe
        sort_fn = lambda df: df.sort_values(by=date_column)

        # Applying log to each value
        log_fn = lambda df: df.assign(**{col: np.log(df[col]) for col in price_columns})

        # Calculating the difference
        diff_fn = lambda df: df.assign(
            **{col: 100 * (df[col] - df[col].shift(1)) for col in price_columns}).reset_index(drop=True)

        return compose(diff_fn, log_fn, sort_fn)(new_df)

    return p, p(dataset)


@curry
def time_split_dataset(df, train_start_date, train_end_date, holdout_end_date, date_col):
    """
    Splits temporal data into a training and testing datasets such that
    all training data comes before the testings set.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame with an Identifier Column and a Date Column.
        The model will be trained to predict the target column
        from the features.

    train_start_date : str
        A date string representing a the starting time of the training data.
        It should be in the same format as the Date Column in `dataset`.
        Inclusive in the train set

    train_end_date : str
        A date string representing a the ending time of the training data.
        This will also be used as the start date of the holdout period.
        It should be in the same format as the Date Column in `dataset`.
        Inclusive in the train set. Exclusive in the test set.

    holdout_end_date : str
        A date string representing a the ending time of the holdout data.
        It should be in the same format as the Date Column in `dataset`.
        Inclusive in the test set.

    date_col : str
        The name of the Date column of `dataset`.


    Returns
    ----------
    train_set : pandas.DataFrame
        The in ID sample and in time training set.

    test_set : pandas.DataFrame
        The out of time testing set.
    """

    train_set = df.copy()[
        (df[date_col] >= train_start_date) & (df[date_col] <= train_end_date)]

    test_set = df.copy()[
        (df[date_col] > train_end_date) & (df[date_col] <= holdout_end_date)]

    return train_set, test_set
