from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class BloodPressureCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        # Функции преобразования — из твоего кода
        def ap_hi(hi):
            if hi < 0:
                return abs(hi)
            elif 1 <= hi <= 2:
                return hi * 100
            elif 3 <= hi <= 25:
                return hi * 10
            elif 270 < hi < 2700:
                return hi // 10
            elif 2700 < hi < 17000:
                return hi // 100
            else:
                return hi
        def ap_lo(lo):
            if lo < 0:
                return abs(lo)
            elif 200 <= lo < 2000:
                return lo // 10
            elif 2000 <= lo < 20000:
                return lo // 100
            else:
                return lo
        X['ap_hi'] = X['ap_hi'].map(ap_hi)
        X['ap_lo'] = X['ap_lo'].map(ap_lo)
        # Переставляем местами при необходимости
        mask = X['ap_hi'] < X['ap_lo']
        X.loc[mask, ['ap_hi', 'ap_lo']] = X.loc[mask, ['ap_lo', 'ap_hi']].values
        return X


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def ag_step(self, ap_sis):
        # Степень артериальной гипертензии по систолическому давлению
        if ap_sis < 140:
            return 0
        elif 140 <= ap_sis < 160:
            return 1
        elif 160 <= ap_sis < 180:
            return 2
        elif ap_sis >= 180:
            return 3
        else:
            return 4

    def transform(self, X):
        X = X.copy()
        # Возраст в годах
        X['years'] = (X['age'] // 365).round().astype('int')
        # Среднее артериальное давление
        X['avrg_ap'] = np.round((2 * X['ap_lo'] + X['ap_hi']) / 3, 1)
        # Составной признак, отражающий зависимость ССЗ
        X['aphi_chol_gluc'] = (X['avrg_ap'] + X['years']) * (X['cholesterol'] + X['gluc'])
        # Признак степени гипертензии
        X['ag_st'] = X['ap_hi'].map(self.ag_step)
        # Составной признак, отражающий зависимость ССЗ 2
        X['ssz_risc'] = (X['years'] + X['avrg_ap'] + X['aphi_chol_gluc'] // 10)
        return X