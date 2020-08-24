import os, csv, json, pandas as pd
import annotate, collect, supervise
from pprint import pprint
import numpy as np

def get_labeled(basedir, annotationdir):
    def read_rows(name):
        fname = os.path.join(basedir, 'csv', name)
        try:
            return list(csv.reader(open(fname)))
        except:
            return list(csv.reader(open(fname, encoding='cp1252')))

    iter_labeled_meta = annotate.get_metadata(basedir, annotationdir)
    iter_all_meta = collect.get_all_metadata(basedir)
    n_unlabeled = 0
    return supervise.make_items(iter_labeled_meta, iter_all_meta, n_unlabeled,
                                read_rows)[0]


def get_column_test(basedir, annotationdir):
    from columns import SupervisedPipeline

    labeled_items = get_labeled(basedir, annotationdir)

    X, y = zip(*SupervisedPipeline.make_records(*zip(*labeled_items)))
    X = pd.DataFrame.from_records(X).set_index('@id').replace([pd.np.nan], 0)
    y = pd.Series(y, index=X.index)

    #     FILTER = 10
    #     c = y.value_counts()
    #     indices_of_non_repeated_labels = y.apply(lambda l: l in set(c[c > FILTER].keys()))
    #     X, y = X[indices_of_non_repeated_labels], y[indices_of_non_repeated_labels]

    return X, y


def get_table_test(basedir, annotationdir):
    from tabletypes import SupervisedPipeline

    labeled_items = get_labeled(basedir, annotationdir)

    X, y = zip(*SupervisedPipeline.make_records(*zip(*labeled_items)))
    X = pd.DataFrame.from_records(list(X)).set_index('@id').replace(
        [pd.np.nan], 0)
    y = pd.Series(y, index=X.index)

    return X, y


# def get_tablepedia_test(basedir, annotationdir):

#     X, y = get_column_test(basedir, annotationdir)

#     X['label'] = y
#     tp_test = X.loc[(X['label'] == 'http://karmaresearch.net/Dataset') |
#                     (X['label'] == 'http://karmaresearch.net/Metric')  |
#                     (X['label'] == 'http://karmaresearch.net/Method')  ]

#     return tp_test.iloc[:, :-1], tp_test['label']
