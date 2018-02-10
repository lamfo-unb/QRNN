import numpy as np
from toolz import curry
import pandas.api.types as ptypes

@curry
def diff_log_pricer(df, price_cols, date_col):
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

    assert ptypes.is_datetime64_any_dtype(df[date_col]), "date column is not of type datetime"
    assert all(ptypes.is_numeric_dtype(df[col]) for col in price_cols), "one or more price column is not numeric"
    assert df.isnull().any().sum() == 0, "NaNs found on the dataset"

    new_df = df.sort_values(by=date_col)
    new_df.index = df.index
    new_df[price_cols] = np.log(new_df[price_cols])
    new_df[price_cols] = 100 * (new_df[price_cols] - new_df[price_cols].shift(1))
    return new_df


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
    assert ptypes.is_datetime64_any_dtype(df[date_col]), "date column is not of type datetime"

    train_set = df[
        (df[date_col] >= train_start_date) & (df[date_col] <= train_end_date)]

    test_set = df[
        (df[date_col] > train_end_date) & (df[date_col] <= holdout_end_date)]

    return train_set, test_set