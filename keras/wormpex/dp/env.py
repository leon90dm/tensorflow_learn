from wormpex.dp.config import *
import pandas as pd
import numpy as np


class OffDataSetEnv(object):
    def __init__(self, file_name):
        self.reward_col = ['last_window_sale_amount']
        self.feature_cols = ['last_window_sale_amount', 'total_sale_amount',
                             'time',
                             'total_pass_rate', 'last_hour_pass_rate',
                             'rest_inventory']
        self.action_col = ['discount']

        self.df = pd.read_csv(file_name)
        self.df.set_index(['date', 'time'], inplace=True)
        self.df = self.df[self.df.discount.notna()]
        self.df.drop([self.df.columns[-1]], axis=1, inplace=True)
        self.df.fillna(NAN_FILL, inplace=True)

        self.data_size = len(self.df.index)
        self.loci = 0

    def reset(self):
        self.loci = np.random.choice(self.data_size)
        return self.loci

    def step(self, action):
        pass


def make():
    return OffDataSetEnv('yinke_noodles_data.csv')

