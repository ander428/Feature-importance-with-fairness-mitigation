# from https://github.com/ResponsiblyAI/responsibly/blob/master/responsibly/dataset/compas/__init__.py
__all__ = ['COMPASDataset']

import numpy as np
import pandas as pd

import abc

class Dataset(abc.ABC):
    """Base class for datasets.

    Attributes
        - `df` - :class:`pandas.DataFrame` that holds the actual data.

        - `target` - Column name of the variable to predict
                    (ground truth)

        - `sensitive_attributes` - Column name of the
                                sensitive attributes

        - `prediction` - Columns name of the
                        prediction (optional)

    """

    @abc.abstractmethod
    def __init__(self, target, sensitive_attributes, prediction=None):
        """Load, preprocess and validate the dataset.

        :param target: Column name of the variable
                    to predict (ground truth)
        :param sensitive_attributes: Column name of the
                                    sensitive attributes
        :param prediction: Columns name of the
                           prediction (optional)
        :type target: str
        :type sensitive_attributes: list
        :type prediction: str
        """

        self.df = self._load_data()

        self._preprocess()

        self._name = self.__doc__.splitlines()[0]

        self.target = target
        self.sensitive_attributes = sensitive_attributes
        self.prediction = prediction

        self._validate()

    def __str__(self):
        return ('<{} {} rows, {} columns'
                ' in which {{{}}} are sensitive attributes>'
                .format(self._name,
                        len(self.df),
                        len(self.df.columns),
                        ', '.join(self.sensitive_attributes)))

    @abc.abstractmethod
    def _load_data(self):
        pass

    @abc.abstractmethod
    def _preprocess(self):
        pass

    @abc.abstractmethod
    def _validate(self):
        # pylint: disable=line-too-long

        assert self.target in self.df.columns,\
            ('the target label \'{}\' should be in the columns'
             .format(self.target))

        assert all(attr in self.df.columns
                   for attr in self.sensitive_attributes),\
            ('the sensitive attributes {{{}}} should be in the columns'
             .format(','.join(attr for attr in self.sensitive_attributes
                              if attr not in self.df.columns)))

        # assert all(attr in SENSITIVE_ATTRIBUTES
        #           for attr in self.sensitive_attributes),\
        # ('the sensitive attributes {} can be only from {}.'  # noqa
        #  .format(self.sensitive_attributes, SENSITIVE_ATTRIBUTES))



COMPAS_PATH = 'data/compas-scores-two-years.csv'


class COMPASDataset(Dataset):
    """ProPublica Recidivism/COMPAS Dataset.

    See :class:`~responsibly.dataset.Dataset` for a description of
    the arguments and attributes.

    References:
        https://github.com/propublica/compas-analysis

    """

    def __init__(self):
        super().__init__(target='is_recid',
                         sensitive_attributes=['race', 'sex'],
                         prediction=['y_pred', 'score_factor',
                                     'score_text'])

    def _load_data(self):
        return pd.read_csv(COMPAS_PATH)

    def _preprocess(self):
        """Perform the same preprocessing as the original analysis.

        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """

        self.df = self.df[(self.df['days_b_screening_arrest'] <= 30)
                          & (self.df['days_b_screening_arrest'] >= -30)
                          & (self.df['is_recid'] != -1)
                          & (self.df['c_charge_degree'] != 'O')
                          & (self.df['score_text'] != 'N/A')]

        self.df['c_jail_out'] = pd.to_datetime(self.df['c_jail_out'])
        self.df['c_jail_in'] = pd.to_datetime(self.df['c_jail_in'])
        self.df['length_of_stay'] = (self.df['c_jail_out']
                                     - self.df['c_jail_in'])

        self.df['score_factor'] = np.where(self.df['score_text']
                                           != 'Low',
                                           'HighScore', 'LowScore')
        self.df['y_pred'] = (self.df['score_factor'] == 'HighScore')

    def _validate(self):
        # pylint: disable=line-too-long
        super()._validate()

        assert len(self.df) == 6172, 'the number of rows should be 6172,'\
                                     ' but it is {}.'.format(len(self.df))
        assert len(self.df.columns) == 56, 'the number of columns should be 56,'\
                                           ' but it is {}.'.format(len(self.df.columns))
