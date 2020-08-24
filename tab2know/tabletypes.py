import os, sys, glob, collections, re
import dpath.util


class KeyWordMatcher:
    def __init__(self, modeldir):
        pass

    def transform(self, meta, rows, verbose=False, **kwargs):
        t = None
        prefix = 'http://karmaresearch.net/'

        caption = meta['dc:title'].lower()

        if 'result' in caption:
            t = prefix + 'Observation'
        elif 'data' in caption:
            t = prefix + 'Input'
        else:
            t = prefix + 'Other'

        dpath.util.new(meta, 'rdf:type', t)
        return meta


import pickle, pandas as pd


def numericalRate(col):
    def numeric(cell):
        return sum(1 for c in cell if c.isnumeric()) / len(cell) if cell else 0

    return pd.np.mean([numeric(cell) for cell in col])


import statistics


def get_numerical_features(bodycells):
    def extract_numbers(cell):
        num = re.findall('[0-9\.]+', cell)
        if num:
            try:
                yield float(num[0])
            except ValueError:
                pass

    numericcells = [num for cell in bodycells for num in extract_numbers(cell)]
    return {
        'sum':
        sum(numericcells) if len(numericcells) > 0 else 0,
        'max':
        max(numericcells) if len(numericcells) > 0 else 0,
        'min':
        min(numericcells) if len(numericcells) > 0 else 0,
        'mean':
        statistics.mean(numericcells) if len(numericcells) > 0 else 0,
        'median':
        statistics.median(numericcells) if len(numericcells) > 0 else 0,
        'stdev':
        statistics.stdev(numericcells) if len(numericcells) > 1 else 0,
        'variance':
        statistics.variance(numericcells) if len(numericcells) > 1 else 0
    }


import re
NUMBERS = re.compile(r'\d+')


def process_text(text):
    return NUMBERS.sub('#', text)


class SupervisedPipeline:
    def __str__(self):
        return f'{self.__class__.__name__}(fname={self.fname})'

    def __init__(self, modeldir, pkl='tabletypes-svm-pipeline.pkl', **kwargs):
        self.fname = os.path.join(modeldir, pkl)
        self.pipeline = pickle.load(open(self.fname, 'rb'))

        import supervise
        self.labelqueries = supervise.load_labelquery_results(modeldir).get(
            'table-type', {})


    @staticmethod
    def make_records(metadatas, tables, unlabeled=False, **kwargs):
        for meta, rows in zip(metadatas, tables):
            nhead = dpath.util.get(meta, 'dialect/headerRowCount',
                                   default=0) or 0
            cols = list(zip(*rows))
            y = meta.get('rdf:type')

            if unlabeled or y:
                numericContent = [numericalRate(col[nhead:]) for col in cols]
                numericContent = pd.np.mean(
                    numericContent) if numericContent else 0

                bodycells = [cell for col in cols for cell in col[nhead:]]
                numfeatures = get_numerical_features(bodycells)

                x = {
                    '@id':
                    meta.get('@id'),
                    'tableCaption':
                    process_text(meta['dc:title']),
                    'headerText':
                    process_text(' '.join(' '.join(col[:nhead])
                                          for col in cols)),
                    'columnText':
                    process_text(' '.join(' '.join(col[nhead:])
                                          for col in cols)),
                    'allText':
                    process_text(meta['dc:title'] + ' '.join(' '.join(col)
                                                             for col in cols)),
                    'numericContent':
                    numericContent,
                    **{k: v
                       for k, v in numfeatures.items()}
                }
                yield x, (y or None)

    def transform(self, meta, rows, **kwargs):
        X, y = zip(
            *SupervisedPipeline.make_records([meta], [rows], unlabeled=True))
        x = pd.DataFrame.from_records(X).set_index('@id')
        x = x.replace([pd.np.inf, -pd.np.inf, pd.np.nan], 0)

        try:
            dpath.util.new(meta, 'rdf:type', self.pipeline.predict(x)[0])
        except Exception as e:
            print(X)
            raise e
        return meta

models = {
    'heuristic': (KeyWordMatcher, {}),
    'supervised': (SupervisedPipeline, {}),
    'supervised-svm': (SupervisedPipeline, {}),
    'supervised-svm-num': (SupervisedPipeline, {
        'pkl': 'tabletypes-numfeatures-svm-pipeline.pkl'
    }),
    'supervised-lr': (SupervisedPipeline, {
        'pkl': 'tabletypes-lr-pipeline.pkl'
    }),
    'supervised-nb': (SupervisedPipeline, {
        'pkl': 'tabletypes-nb-pipeline.pkl'
    }),
    'supervised-1k-lq': (SupervisedPipeline, {
        'n_unlabeled': 1000,
        'labelqueries_exclude': []
    }),
    'supervised-10k-lq': (SupervisedPipeline, {
        'n_unlabeled': 10000,
        'labelqueries_exclude': []
    }),

    #     'supervised (500 self-trained, prob > 0)': (SupervisedPipeline, {'n_unlabeled': 500, 'self_train_threshold': 0}),
    #     'supervised (500 self-trained, prob > .5)': (SupervisedPipeline, {'n_unlabeled': 500, 'self_train_threshold': .5}),
    #     'supervised (500 self-trained, prob > .75)': (SupervisedPipeline, {'n_unlabeled': 500, 'self_train_threshold': .75}),

    #     'supervised (1000 self-trained, prob > 0)': (SupervisedPipeline, {'n_unlabeled': 1000, 'self_train_threshold': 0}),
    #     'supervised (1000 self-trained, prob > .5)': (SupervisedPipeline, {'n_unlabeled': 1000, 'self_train_threshold': .5}),
    #     'supervised (1000 self-trained, prob > .75)': (SupervisedPipeline, {'n_unlabeled': 1000, 'self_train_threshold': .75}),
}

default_model = lambda x: (SupervisedPipeline, {'supdir': x})
