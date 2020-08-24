import os, sys, json, warnings, logging as log
import pandas as pd, tqdm, dpath
import annotate, collect
from pprint import pprint


def make_items(iter_labeled_meta, iter_all_meta, n_unlabeled, read_rows):
    '''Generate metadata from gold-standard and unlabled'''
    
    labeled_items = [(meta, read_rows(meta['url']))
                     for meta in iter_labeled_meta]
    annotated_table_urls = set([meta['url'] for meta, _ in labeled_items])

    unlabeled_meta = []
    for meta in iter_all_meta:
        if (n_unlabeled is not None) and len(unlabeled_meta) >= n_unlabeled:
            break
        if meta['url'] not in annotated_table_urls:
            unlabeled_meta.append(meta)
    unlabeled_items = [(meta, read_rows(meta['url']))
                       for meta in unlabeled_meta]

    return labeled_items, unlabeled_items


def make_labelquery(args):
    querytype, template, slots, value, templates, namespace, kbdomain, name = args
    return querytype, name, annotate.make_labelquery(*args)


def parallel_query(labelqueries,
                   templates,
                   namespace,
                   kbdomain,
                   max_workers=1):
    import tqdm, multiprocessing
    with multiprocessing.Pool(max_workers) as p:
        stream_args = [(q['label'], q['template'], q['slots'], q['value'],
                        templates, namespace, kbdomain, name)
                       for name, q in labelqueries.items()]
        t = len(stream_args)
        #         yield from tqdm.tqdm(p.imap_unordered(make_labelquery, stream_args), total=t)
        yield from p.imap_unordered(make_labelquery, stream_args)


def cache_labelquery_results(modeldir,
                             namespace,
                             kbdomain,
                             selected_queries=[],
                             results_fname=None,
                             parallel=False,
                             verbose=False):
    labelqueries, templates = annotate.load_labelqueries_templates(modeldir)
    if not results_fname:
        os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'),
                    exist_ok=True)
        results_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                     'results.json')

    labelquery_results = load_labelquery_results(modeldir,
                                                 results_fname=results_fname)
    l = len(labelqueries)
    if parallel:
        if selected_queries:
            labelqueries = {
                name: q
                for name, q in labelqueries.items() if name in selected_queries
            }

        lqs = parallel_query(labelqueries,
                             templates,
                             namespace,
                             kbdomain,
                             max_workers=parallel)
        for qt, name, lq in lqs:
            labelquery_results.setdefault(qt, {})[name] = lq
    else:
        for i, (name, q) in enumerate(labelqueries.items()):
            if selected_queries and (name not in selected_queries):
                continue
            lq = annotate.make_labelquery(q['label'],
                                          q['template'],
                                          q['slots'],
                                          q['value'],
                                          templates,
                                          namespace,
                                          kbdomain=kbdomain,
                                          name=name)
            if verbose:
                print(len(lq.transformations), 'results')
            labelquery_results.setdefault(q['label'], {})[name] = lq

    with open(results_fname, 'w') as fw:
        results_json = {
            label: {name: vars(lq)
                    for name, lq in lqs.items()}
            for label, lqs in labelquery_results.items()
        }
        json.dump(results_json, fw, indent=2)

    with open(results_fname.replace('.json', '.stats.json'), 'w') as fw:
        results_json = {
            name: len(lq.transformations)
            for label, lqs in labelquery_results.items()
            for name, lq in lqs.items()
        }
        json.dump(results_json, fw, indent=2)

    return labelquery_results


def load_labelquery_results(modeldir, results_fname=None):
    typed_labelqueries = {}

    if not results_fname:
        os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'),
                    exist_ok=True)
        results_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                     'results.json')
    if os.path.exists(results_fname):
        typed_labelqueries = json.load(open(results_fname))
        for lq_type, labelqueries in typed_labelqueries.items():
            for name, lq_params in labelqueries.items():
                labelqueries[name] = annotate.LabelQuery(**lq_params)
    return typed_labelqueries


def transform_all(labelqueries, unlabeled_items, model, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        id_items = {m['@id']: (m, r) for m, r in unlabeled_items}

        lX = []
        lq_labels = {}
        l = len(labelqueries)
        for i, (name, lq) in enumerate(labelqueries.items()):
            print(f'Transforming using query {name:>4s} [{i+1:3d}/{l:3d}] ...',
                  end='\r',
                  file=sys.stderr)

            # Get corresponding metadata for query results
            selected_items = [
                id_items[i] for i in lq.transformations if i in id_items
            ]
            transformed_items = tuple(
                zip(*[(lq.transform(m, r, **kwargs), r)
                      for m, r in selected_items]))

            if transformed_items:
                recs = tuple(
                    zip(*model.__class__.make_records(*transformed_items)))
                if recs:
                    qlX, qly = recs
                    qlX = pd.DataFrame.from_records(list(qlX)).set_index('@id')
                    lX.append(qlX)
                    lq_labels[name] = pd.Series(qly, index=qlX.index)
        print(file=sys.stderr)
        lX = pd.concat(lX).drop_duplicates().replace([pd.np.nan], 0)

        L = pd.DataFrame(index=lX.index)  # rows: n_labelqueries x cols: labels
        for lqname, qly in lq_labels.items():
            L[lqname] = qly

    return lX, L


def get_query_labelings(labeled_metas, labelqueries):
    item_query_label = {}
    for meta in labeled_metas:
        for qid, lq in labelqueries.items():
            for p, v in lq.transformations.get(meta['@id'], {}).items():
                if v:
                    item_query_label.setdefault((meta['url'], p), {})[qid] = v
    L = pd.DataFrame.from_dict(item_query_label, orient='index')
    return L


def get_true_labelings(labeled_metas, eval_path):
    item_truelabel = {}
    for meta in labeled_metas:
        for p, v in dpath.util.search(meta, eval_path, yielded=True):
            if not meta.get('karma:isBad'):
                item_truelabel[(meta['url'], p)] = v or None
    return pd.Series(item_truelabel)


def to_numbered(L, y_true):
    # Replace class labels by integers (nan => -1)
    label_values = sorted(set(['']) | set(y_true.fillna('').values.flatten()))
    value_repl = {v: i - 1 for i, v in enumerate(label_values)}
    value_repl['UNK'] = -1

    L = L.fillna(-1).replace(value_repl).astype('int32').values
    y_true = y_true.fillna(-1).replace(value_repl).astype('int32').values
    return L, y_true


def from_numbered(L, y_true, y_pred):
    label_values = sorted(set(['']) | set(y_true.fillna('').values.flatten()))
    repl_value = {i - 1: v for i, v in enumerate(label_values)}
    y_pred = pd.Series(y_pred, index=L.index)
    return y_pred.replace(repl_value).replace({'': pd.np.nan})


def labelmodel_predict(L_train, y_true, L_test, return_probs=False, **kwargs):
    kwargs.setdefault('n_epochs', 500)
    kwargs.setdefault('log_freq', 100)

    from snorkel.labeling.model import LabelModel
    n = len(set(y_true[~y_true.isna()].values))
    log.info('y_true values: %s', set(y_true[~y_true.isna()].values))
    label_model = LabelModel(cardinality=n, verbose=True)

    L_train_val = set(L_train.values.flatten())
    y_true_val = set(y_true.values.flatten())
    log.info('Values in L_train but not y_true: %s', L_train_val - y_true_val)
    log.info('Values in y_true but not L_train: %s', y_true_val - L_train_val)

    L_train, Y_dev = to_numbered(L_train, y_true)

    log.info('L_train values: %s, %s', set(L_train.flatten()), type(L_train))
    log.info('Y_dev values: %s, %s', set(Y_dev.flatten()), type(Y_dev))
    log.info('kwargs: %s', kwargs)

    label_model.fit(L_train=L_train, Y_dev=Y_dev[Y_dev != -1], **kwargs)

    y_pred = label_model.predict(to_numbered(L_test, y_true)[0],
                                 return_probs=return_probs)

    if return_probs:
        y_pred, y_score = y_pred
    y_pred = from_numbered(L_test, y_true, y_pred)
    return (y_pred, y_score) if return_probs else y_pred


def save_query_analysis(modeldir, labeled_metas, labelquery_type):
    os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'), exist_ok=True)
    results_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                 'gold.json')
    typed_labelqueries = load_labelquery_results(modeldir,
                                                 results_fname=results_fname)

    eval_path = annotate.labelquery_types[labelquery_type]['eval_path']
    y_true = get_true_labelings(labeled_metas, eval_path)
    L = get_query_labelings(labeled_metas, typed_labelqueries[labelquery_type])
    L = L.reindex(index=y_true.index)

    unk_value = annotate.labelquery_types[labelquery_type]['UNK']
    L['-1'] = unk_value
    L = L.replace({'UNK': unk_value})
    y_true = y_true.replace({None: unk_value}).fillna(unk_value)

    for modelname in ['-majority', '']:
        import importlib
        if importlib.util.find_spec("snorkel") is None:
            y_pred = L.mode(axis=1)[0]
            modelname = ''
        else:
            if modelname:
                y_pred = L.mode(axis=1)[0]
            else:
                log.info(f'Fitting label model for {labelquery_type}...')
                y_pred = labelmodel_predict(L, y_true, L)
        name = f'{labelquery_type}{modelname}'

        y = pd.DataFrame({'true': y_true, 'pred': y_pred}).fillna(unk_value)

        y.to_csv(
            os.path.join(modeldir, 'labelqueries', 'cache',
                         f'{name}-labeling.csv'))
        target_names = [
            x.split('/')[-1] for x in sorted(set(y.true) | set(y.pred))
        ]

        import analyse
        report = analyse.pd_report(y.true, y.pred, target_names)
        log.info('Accuracy (%s): %.2f', name, report.loc['accuracy',
                                                         'f1-score'])
        report.index.name = 'label'
        report.to_csv(
            os.path.join(modeldir, 'labelqueries', 'cache',
                         f'{name}-report.csv'))
        confusion = analyse.pd_confusion(y.true,
                                         y.pred,
                                         target_names,
                                         stack=True)
        confusion.to_csv(
            os.path.join(modeldir, 'labelqueries', 'cache',
                         f'{name}-confusion.csv'))


def sample_match_dist(y, y_match, subsample=1, **kwargs):
    y, y_match = pd.Series(y), pd.Series(y_match)
    n = len(y)
    y_count = y_match.value_counts()
    y_count = y_count**subsample
    y_frac = y_count / sum(y_count)

    missing_classes = set(v for v, f in y_frac.items() if not any(y == v))
    if missing_classes:
        raise Exception(
            f'Cannot match distribution: no occurrence of classes: ',
            list(missing_classes))

    samples = [
        y[y == v].sample(int(n * f), replace=(n * f > sum(y == v)), **kwargs)
        for v, f in y_frac.items()
    ]
    return pd.concat(samples)


def frame(model, items, unlabeled=False):
    X, y = zip(
        *model.__class__.make_records(*zip(*items), unlabeled=unlabeled))
    X = pd.DataFrame.from_records(list(X))
    X = X.set_index('@id').replace([pd.np.nan, pd.np.inf, -pd.np.inf], 0)
    y = pd.Series(y, index=X.index)
    return X, y


def ablate(labeled_items,
           labelqueries,
           focus_queries,
           unlabeled_items,
           model,
           n_repeats,
           cache=None,
           verbose=False):
    # Report the effect of removing focus_query from training
    vec, clf = model.pipeline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        X, y = frame(model, labeled_items)

        lX, L = transform_all(labelqueries, unlabeled_items, model)
        ly = L.mode(axis=1)[0]  # Most frequent label value

    print(f'Using {X.shape[0]} labeled and {lX.shape[0]} query-labeled items',
          file=sys.stderr)

    vec.fit(pd.concat([X, lX]))
    print(f'Model has {len(vec.get_feature_names())} features')

    import analyse

    def report(_X, _y, focus_label):

        _r = sample_match_dist(_y, y).index
        y_pred = clf.fit(vec.transform(_X.loc[_r]),
                         _y.loc[_r]).predict(vec.transform(X))
        rep = analyse.pd_report(y, y_pred).loc[focus_label]
        return rep

    def yield_ablations(focus_query, focus_label):
        abl = L[(q for q in L.columns
                 if q != focus_query)]  # Remove focus query
        ay = abl.mode(axis=1)[0]  # Most frequent label value
        aX, ay = lX[~ay.isna()], ay[~ay.isna()]

        if focus_label in set(ay):
            for i in range(n_repeats):
                d = (report(lX, ly, focus_label) - report(aX, ay, focus_label))
                if verbose:
                    name = f'Query {focus_query} ({focus_label.split("/")[-1]:>25s})'
                    print(name,
                          ' '.join(f'{k}:{v:.2f}' for k, v in d.items()),
                          file=sys.stderr)

                yield d

    scores = {}
    with tqdm.tqdm(total=(len(focus_queries) * n_repeats)) as pbar:
        for focus_query in focus_queries:
            if focus_query in labelqueries:
                try:
                    transformations = labelqueries[focus_query].transformations
                    focus_label = next((v for vs in transformations.values()
                                        for v in vs.values()), None)

                    if verbose:
                        all_tableids = set(m['@id']
                                           for m, r in unlabeled_items)
                        l = len(all_tableids & set(transformations))
                        print(
                            f'Query {focus_query} labels {l} / {len(all_tableids)} tables'
                        )
                    pbar.set_description(
                        f'Query {focus_query} ({str(focus_label).split("/")[-1]:>25s})'
                    )

                    abl = yield_ablations(focus_query, focus_label)
                    if abl is not None:
                        diffs = [d for d in abl if not pbar.update(1)]
                        if diffs:
                            scores[focus_query] = pd.DataFrame(
                                diffs).describe().loc[['mean', 'std']].round(2)
                        continue

                except Exception as e:
                    print(e, file=sys.stderr)
                pbar.update(n_repeats)

        pbar.update(1)

    return scores
