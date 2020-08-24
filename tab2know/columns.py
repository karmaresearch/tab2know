import os, sys, glob, collections, re
import json, csv
import dpath.util, rdflib
from rdflib import Graph
from pprint import pprint
import pandas as pd
import pickle, statistics

import re
NUMBERS = re.compile(r'\d+')


def numericalRate(col):
    def numeric(cell):
        return sum(1 for c in cell if c.isnumeric()) / len(cell) if cell else 0

    return pd.np.nanmean([numeric(cell) for cell in col])


def process_text(text):
    return NUMBERS.sub('#', text)


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



class LabelMatchingModel:
    _TOKENS = re.compile(r"(?u)\b\w\w+\b")

    def __init__(self, modeldir):

        self.g = rdflib.Graph()
        for fname in glob.glob(os.path.join(modeldir, '*.ttl')):
            self.g.parse(fname, format="n3")

        label = rdflib.term.URIRef(
            'http://www.w3.org/2000/01/rdf-schema#label')
        altlabel = rdflib.term.URIRef(
            'http://www.w3.org/2004/02/skos/core#altLabel')

        uri_labels = collections.defaultdict(set)
        for s, p, o in list(self.g.triples(
            (None, label, None))) + list(self.g.triples(
                (None, altlabel, None))):
            label = str(o).lower()
            uri = str(s)
            uri_labels[uri].add(label)

        self.label_uri_score = collections.defaultdict(collections.Counter)
        for uri, labels in uri_labels.items():
            for l in labels:
                self.label_uri_score[l][uri] += 1 / len(labels)
        for label, uri_score in self.label_uri_score.items():
            t = sum(uri_score.values())
            self.label_uri_score[label] = collections.Counter(
                {k: v / t
                 for k, v in uri_score.items()})

    def match(self, cells, verbose=False):
        match_score = collections.Counter()
        for cell in cells:
            cell = cell.lower()
            match_score.update(self.label_uri_score.get(cell, {}))
            for tok in self._TOKENS.findall(cell):
                match_score.update(self.label_uri_score.get(tok, {}))

        if verbose and match_score:
            print(cells, *match_score.items())
        for m, c in match_score.most_common(1):
            return m

    def transform(self, meta, rows, verbose=False, **kwargs):
        n_head = dpath.util.get(meta, 'dialect/headerRowCount', default=0) or 0
        header = rows[:n_head]

        for ci, hcol in enumerate(zip(*header)):
            match = self.match(hcol, verbose=verbose)
            if match:
                dpath.util.new(meta, f'tableSchema/columns/{ci}/propertyUrl',
                               match)

        return meta


class SupervisedPipeline:
    def __str__(self):
        return f'{self.__class__.__name__}(fname={self.fname})'

    def __init__(self,
                 modeldir,
                 use_hier=False,
                 supdir='columns-svm-pipeline.pkl',
                 heirdir='columns-svm-hierarchy-pipeline.pkl',
                 **kwargs):

        self.fname = os.path.join(modeldir,
                                  supdir) if not use_hier else os.path.join(
                                      modeldir, heirdir)
        self.pipeline = pickle.load(open(self.fname, 'rb'))
        self.modeldir = modeldir

        import supervise
        self.labelqueries = supervise.load_labelquery_results(modeldir).get(
            'column-property', {})

    @staticmethod
    def make_records(metadatas, tables, unlabeled=False, **kwargs):
        for meta, rows in zip(metadatas, tables):
            nhead = dpath.util.get(meta, 'dialect/headerRowCount',
                                   default=0) or 0
            has_annotations = any(v for d, v in dpath.util.search(
                meta, f'tableSchema/columns/*/propertyUrl', yielded=True))

            cols = list(zip(*rows))
            for ci, col in enumerate(cols):
                if has_annotations:
                    default = 'UNK'
                else:
                    default = None
                y = dpath.util.get(meta,
                                   f'tableSchema/columns/{ci}/propertyUrl',
                                   default=default)
                y = y or default

                if unlabeled or y:
                    other_headercells = [
                        col2[:nhead] for ci2, col2 in enumerate(cols)
                        if ci2 != ci
                    ]
                    headercells = col[:nhead]
                    bodycells = col[nhead:]
                    numfeatures = get_numerical_features(bodycells)
                    # add features
                    x = {
                        '@id':
                        meta.get('@id') + f'#col={ci+1}',
                        'tableCaption':
                        process_text(meta['dc:title']),
                        'headerText':
                        process_text(' '.join(headercells)),
                        'otherHeaderText':
                        process_text(' '.join(' '.join(cell)
                                              for cell in other_headercells)),
                        'columnText':
                        process_text(' '.join(bodycells)),
                        'numericContent':
                        numericalRate(bodycells),
                        'numSum':
                        numfeatures['sum'],
                        'numMax':
                        numfeatures['max'],
                        'numMin':
                        numfeatures['min'],
                        'numMean':
                        numfeatures['mean'],
                        'numMedian':
                        numfeatures['median'],
                        'stdev':
                        numfeatures['stdev'],
                        'variance':
                        numfeatures['variance']
                    }
                    yield x, y

    @staticmethod
    def build_adjacency_lists(modeldir='models/'):
        from sklearn_hierarchical_classification.constants import ROOT
        g = Graph()
        for fname in glob.glob(os.path.join(modeldir, '*.ttl')):
            g.parse(fname, format="n3")

        adj_list = {ROOT: []}
        r_adj_list = {}
        for child, _, parent in g.triples(
            (None, rdflib.RDFS.subClassOf, None)):
            if str(parent).startswith("ub"):
                continue
            parent, child = str(parent), str(child)
            if parent == 'http://www.w3.org/2002/07/owl#Thing':
                parent = ROOT
            adj_list.setdefault(parent, [])
            adj_list[parent].append(child)
            r_adj_list.setdefault(child, [])
            r_adj_list[child].append(parent)

        return adj_list, r_adj_list

    @staticmethod
    def reduce_label_tree(adj_list, r_adj_list):
        fname = 'annotations/venue-filtered-sample-200/all_metadata.jsonl'
        csvdir = 'data/extracted/selection/s2/csv/'
        metadatas = [json.loads(line) for line in open(fname)]
        csvs = [
            list(csv.reader(open(csvdir + meta['url']))) for meta in metadatas
        ]
        _, y = zip(*SupervisedPipeline.make_records(metadatas, csvs))
        y = pd.Series(y)
        from sklearn_hierarchical_classification.constants import ROOT

        reduced_tree = {ROOT: ['UNK']}
        for label in set(y.values):
            if label == 'UNK':
                continue
            child = label
            parent = r_adj_list[label][0]

            while True:
                reduced_tree.setdefault(parent, [])
                if not child in reduced_tree[parent]:
                    reduced_tree[parent].append(child)
                if parent == ROOT:
                    break
                child = parent
                parent = r_adj_list[parent][0]

        return reduced_tree

    def transform(self, meta, rows, **kwargs):
        X, y = zip(
            *SupervisedPipeline.make_records([meta], [rows], unlabeled=True))
        X = pd.DataFrame.from_records(X).set_index('@id')
        X = X.replace([pd.np.inf, -pd.np.inf, pd.np.nan], 0)
        for ci, prop in enumerate(self.pipeline.predict(X)):
            if prop != 'UNK':
                dpath.util.new(meta, f'tableSchema/columns/{ci}/propertyUrl',
                               prop)
        return meta


models = {
    'heuristic': (LabelMatchingModel, {}),
    'supervised': (SupervisedPipeline, {}),
    'supervised-svm': (SupervisedPipeline, {}),
    'supervised-svm-num': (SupervisedPipeline, {
        'supdir': 'columns-svm-numfeatures-pipeline.pkl'
    }),
    'supervised-lr': (SupervisedPipeline, {
        'supdir': 'columns-lr-pipeline.pkl'
    }),
    'supervised-nb': (SupervisedPipeline, {
        'supdir': 'columns-nb-pipeline.pkl'
    }),

    #     'supervised (100 self-trained)': (SupervisedPipeline, {'n_unlabeled': 100, 'self_train_threshold': .5}),
    #     'supervised (1000 self-trained)': (SupervisedPipeline, {'n_unlabeled': 1000, 'self_train_threshold': .5}),
    #     'supervised (10000 self-trained)': (SupervisedPipeline, {'n_unlabeled': 1000, 'self_train_threshold': .5}),

    #     'hierarchical': (SupervisedPipeline, {'use_hier': True}),
    #     'hierarchical (100 self-trained)': (SupervisedPipeline, {'use_hier': True, 'n_unlabeled': 100, 'self_train_threshold': .5}),
    #     'hierarchical (1000 self-trained)': (SupervisedPipeline, {'use_hier': True, 'n_unlabeled': 1000, 'self_train_threshold': .5}),
    #     'hierarchical (10000 self-trained)': (SupervisedPipeline, {'use_hier': True, 'n_unlabeled': 1000, 'self_train_threshold': .5}),
}

default_model = lambda x: (SupervisedPipeline, {'supdir': x})
