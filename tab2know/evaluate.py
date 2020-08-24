import os, sys, json, csv, copy, pickle, logging as log
import networkx as nx
import pandas as pd
import numpy as np
import dpath.util
import scipy.sparse


def get_confident_predictions(clf, X, exclude_class=None, threshold=0):
    prob = pd.DataFrame(clf.predict_proba(X))
    label_predicted = prob.idxmax(axis=1)

    if exclude_class in clf.classes_:
        exclude_class_i = list(clf.classes_).index(exclude_class)
        n_excluded = (label_predicted == exclude_class_i).sum()

        log.info('n_total: %s n_excluded: %s', len(X), n_excluded)
        index_excluded = (label_predicted != exclude_class_i)
    else:
        index_excluded = prob.index
    label_rank = prob.max(axis=1)[index_excluded].rank(pct=True)
    over_threshold = (label_rank > threshold)
    items = prob.index[index_excluded][over_threshold]

    return X.iloc[items], label_predicted[items].apply(
        lambda x: clf.classes_[x])


def cross_validate(model_name,
                   model,
                   labeled_items,
                   scoring_func,
                   n_splits=5,
                   unlabeled_items=[],
                   exclude_class='UNK',
                   verbose=False,
                   **kwargs):

    import supervise
    X, y = supervise.frame(model, labeled_items)

    # Filter rare classes for cross-valiation
    c = y.value_counts()
    indices_of_non_repeated_labels = y.apply(
        lambda l: l in set(c[c > n_splits].keys()))
    X, y = X[indices_of_non_repeated_labels], y[indices_of_non_repeated_labels]

    # Create unlabeled records
    if unlabeled_items:
        if 'self_train_threshold' in kwargs:
            X_unlabeled, y_unlabeled = supervise.frame(unlabeled_items,
                                                       unlabeled=True)

            vec = model.pipeline[0].fit(pd.concat([X, X_unlabeled]))
            X, X_unlabeled = vec.transform(X), vec.transform(X_unlabeled)

        if 'labelqueries_exclude' in kwargs:
            import supervise
            lX, L = supervise.transform_all(model.labelqueries,
                                            unlabeled_items, model)

            ly = L.mode(axis=1)[0]  # Most frequent label value
            r = supervise.sample_match_dist(ly, y).index
            X_qlabeled = lX.loc[r]
            y_qlabeled = ly.loc[r]

            vec = model.pipeline[0].fit(pd.concat([X, X_qlabeled]))
            X, X_qlabeled = vec.transform(X), vec.transform(X_qlabeled)
    else:
        vec = model.pipeline[0].fit(X)
        X = vec.transform(X)
    y = np.array(y)

    # Split dataset for cross-validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for isplit, (train_index, test_index) in enumerate(skf.split(X, y)):
        log.info(
            f'Evaluating {model_name} ( Fold number {isplit:2d}/{n_splits:2d})...'
        )

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # If we have unlabeled data, do semi-supervision
        if unlabeled_items:
            if 'self_train_threshold' in kwargs:
                # Do self-training
                model.pipeline[-1].fit(X_train, y_train)
                # Get confident predictions
                X_conf, y_conf = get_confident_predictions(
                    model.pipeline[-1],
                    X_unlabeled,
                    exclude_class=exclude_class,
                    threshold=kwargs['self_train_threshold'])
                # Concatenate training sets with extra data
                X_train = scipy.sparse.vstack([X_train, X_conf])
                y_train = list(y_train) + list(y_conf)

            if 'labelqueries_exclude' in kwargs:
                # Concatenate training sets with extra data
                X_train = scipy.sparse.vstack([X_train, X_qlabeled])
                y_train = list(y_train) + list(y_qlabeled)

        # Fit the model
        model.pipeline[-1].fit(X_train, y_train)

        scores = scoring_func(model.pipeline[-1],
                              X_test,
                              y_test,
                              exclude_class=exclude_class,
                              verbose=verbose)
        if unlabeled_items:
            scores['n_unlabeled'] = len(unlabeled_items)
            if 'self_train_threshold' in kwargs:
                scores['n_selftrain'] = X_conf.shape[0]
            if 'labelqueries_exclude' in kwargs:
                scores['n_querylabeled'] = X_qlabeled.shape[0]

        yield scores


def yield_predicted_items(name, gold_items, op_func, var_path, verbose=False):
    for i, (meta_gold, rows) in enumerate(gold_items):
        if all(v in [None, ''] for _, v in dpath.util.search(
                meta_gold, var_path, yielded=True)):
            continue
        meta_pred = copy.deepcopy(meta_gold)
        dpath.util.set(meta_pred, var_path, None)
        meta_pred = op_func(meta_pred, rows)

        gold = {
            p: v
            for p, v in dpath.util.search(meta_gold, var_path, yielded=True)
            if v not in [None, '']
        }
        pred = {
            p: v
            for p, v in dpath.util.search(meta_pred, var_path, yielded=True)
            if v not in [None, '']
        }
        for p in set(gold) | set(pred):
            gold_v = gold.get(p)
            pred_v = pred.get(p)

            if (gold_v != pred_v) and verbose:
                log.info(f'Evaluating model "{name}"')
                log.info(f"{meta_gold['url']}, {p}")
                log.info(f'gold: {gold_v} pred: {pred_v}')

                df = pd.DataFrame(rows)
                log.info(df.shape)
                log.info(df.head().to_string(line_width=None))
                log.info(meta_gold.get('dc:title'))
                op_func(meta_pred, rows, verbose=True)

            yield {(meta_gold['url'], p): {'gold': gold_v, 'pred': pred_v}}


def pipeline_accuracy(pipeline, X, y_true, **kwargs):
    y_pred = pipeline.predict(X)
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    return accuracy(y_true, y_pred)


def pipeline_precision_recall_f1(pipeline, X, y_true, **kwargs):
    y_pred = pipeline.predict(X)
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    exclude_class = kwargs.get('exclude_class')
    if exclude_class:
        y_true = y_true.replace(exclude_class, pd.np.nan)
        y_pred = y_pred.replace(exclude_class, pd.np.nan)
    if kwargs.get('verbose'):
        log.info('accuracy: %d / %d' % ((y_pred == y_true).sum(), len(y_pred)))
        log.info('Top predicted classes:')
        log.info(pd.Series(y_pred).value_counts().head())
    return precision_recall_f1(y_true, y_pred)


modelf = '/path/to/models.pkl'
G = pickle.load(open(modelf, 'rb')).steps[-1][-1].graph_


def pipeline_hierarchical_precision_recall_f1(pipeline, X, y_true, **kwargs):
    y_pred = pipeline.predict(X)
    exclude_class = kwargs.get('exclude_class')
    if exclude_class:
        no_exclude = (y_true != exclude_class)
        y_true = y_true[no_exclude]
        y_pred = y_pred[no_exclude]
    return h_precision_recall_f1(y_true, y_pred)


from sklearn.metrics import accuracy_score


def accuracy(y_true, y_pred):
    return {
        'items': len(y_true),
        'correct': accuracy_score(y_true, y_pred, normalize=False),
        'accuracy': accuracy_score(y_true, y_pred),
    }


from sklearn.metrics import precision_recall_fscore_support


def precision_recall_f1(y_true, y_pred):
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    y_true[~y_true.astype(bool)] = pd.np.nan
    y_pred[~y_pred.astype(bool)] = pd.np.nan
    is_true, is_pred = (~y_true.isna()), (~y_pred.isna())

    tp = sum(y_true[is_true & is_pred] == y_pred[is_true & is_pred])
    fp = sum(y_true[is_pred] != y_pred[is_pred])
    fn = sum(y_true[is_true] != y_pred[is_true])

    p = tp / (tp + fp) if (tp + fp) else (0. if tp else 1.)
    r = tp / (tp + fn) if (tp + fn) else 0.
    f = 2 * (p * r) / (p + r) if p and r else 0.
    return {
        'items': len(y_true),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': p,
        'recall': r,
        'f1': f,
    }


from sklearn_hierarchical_classification.metrics import *


def h_precision_recall_f1(y_true, y_pred):
    import warnings
    warnings.filterwarnings("ignore")  # for `heuristic` model

    y_true, y_pred = np.array([y_true]).T, np.array([y_pred]).T

    #     label_graph = nx.DiGraph(nx.read_adjlist("models/class_hierarchy.graph"))

    with multi_labeled(y_true, y_pred, G) as (y_true, y_pred_, graph_):
        h_precision = h_precision_score(y_true, y_pred_, graph_)
        h_recall = h_recall_score(y_true, y_pred_, graph_)
        h_fbeta = h_fbeta_score(y_true, y_pred_, graph_)
    return {
        'items': len(y_true),
        'h_precision': h_precision,
        'h_recall': h_recall,
        'h_fbeta': h_fbeta,
    }
