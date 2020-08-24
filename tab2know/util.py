from sklearn.preprocessing import Normalizer
import os, csv, pandas as pd
import annotate, collect, supervise


class LabeledNormalizer(Normalizer):
    def fit(self, X, *args, **kwargs):
        try:
            self.names = X.columns
        except:
            self.names = [str(i) for i in range(X.shape[1])]
        return super().fit(X, *args, **kwargs)

    def get_feature_names(self):
        return self.names
