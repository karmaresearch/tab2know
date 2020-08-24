#!/usr/bin/env python3

import os, sys, json, csv, argh
from util import *
import tqdm
import logging as log


def set_verbosity(v):
    if v:
        log.basicConfig(level=log.INFO)
    else:
        loglevel = getattr(log, os.environ.get('LOGLEVEL', ''), log.WARNING)
        logfile = os.environ.get('LOGFILE', None)
        log.basicConfig(filename=logfile, level=loglevel)


import graph


def process_meta(meta,
                 output,
                 structure_model=None,
                 type_model=None,
                 column_model=None,
                 indent=None):
    name = meta['url']
    fname = os.path.join(basedir, 'csv', name)
    try:
        rows = list(csv.reader(open(fname)))
    except:
        try:
            rows = list(csv.reader(open(fname, encoding='cp1252')))
        except:
            rows = []

    if rows:
        meta = structure_model.transform(meta, rows)

        if modeldir:
            meta = type_model.transform(meta, rows)
            meta = column_model.transform(meta, rows)

        if output == 'triples':
            g = graph.convert(json.dumps(meta), read_rows)
            return g.serialize(format=fmt).decode('utf8')
        elif output == 'pred-triples':
            return graph.convert_predictions(meta)
        elif output == 'metadata':
            return json.dumps(meta, indent=indent)


@argh.arg('--indent', '-i', type=int)
def run(
    basedir: 'Directory of PDF extractions',
    annotationdir: 'Annotation Directory' = None,
    modeldir: 'Model directory' = None,
    output: 'Output type (metadata/triples)' = 'triples',
    indent: 'Metadata JSON indentation' = None,
    fmt: 'Triples format' = 'nt',
    paper_prefix: 'Paper URI prefix' = '',
    structure_model: 'Structure prediction approach' = 'heuristic',
    type_model: 'Table type prediction approach' = 'heuristic',
    column_model: 'Column prediction approach' = 'heuristic',
    verbose=False,
):
    """Run pipeline"""
    set_verbosity(verbose)

    def read_rows(name):
        fname = os.path.join(basedir, 'csv', name)
        try:
            return list(csv.reader(open(fname)))
        except:
            try:
                return list(csv.reader(open(fname, encoding='cp1252')))
            except:
                return []

    if output == 'triples':
        import graph

        def make_output(metadata, read_rows):
            g = graph.convert(json.dumps(metadata), read_rows)
            return g.serialize(format=fmt).decode('utf8')
    elif output == 'pred-triples':
        import graph

        def make_output(metadata, read_rows):
            return graph.convert_predictions(metadata)
    elif output == 'metadata':

        def make_output(metadata, read_rows):
            return json.dumps(metadata, indent=indent)
    else:
        raise Exception(f'Output parameter {output} is not valid')

    import structure
    model, kwargs = structure.models[structure_model]
    structure_model = model(modeldir, **kwargs)

    if modeldir:
        import tabletypes
        model, kwargs = tabletypes.models.get(
            type_model, tabletypes.default_model(type_model))
        type_model = model(modeldir, **kwargs)

        import columns
        model, kwargs = columns.models.get(column_model,
                                           columns.default_model(column_model))
        column_model = model(modeldir, **kwargs)

    if annotationdir:
        # Get metadata for annotations
        import annotate
        for meta in tqdm.tqdm(annotate.get_metadata(basedir, annotationdir)):
            log.info("Processing table %s", meta['url'])
            yield make_output(meta, read_rows)
    else:
        # Predict metadata for table extractions using different models
        import collect
        total = len(os.listdir(os.path.join(basedir, 'csv')))
        for meta in tqdm.tqdm(collect.get_all_metadata(
                basedir, paper_prefix=paper_prefix),
                              total=total):
            name = meta['url']
            rows = read_rows(name)

            if rows:
                meta = structure_model.transform(meta, rows)

                if modeldir:
                    meta = type_model.transform(meta, rows)
                    meta = column_model.transform(meta, rows)

                yield make_output(meta, read_rows)


@argh.arg('-o', '--only-selected-models', nargs='*')
def eval(basedir: 'Directory of PDF extractions',
         annotationdir: 'Annotation Directory',
         modeldir: 'Model Directory',
         component: 'Pipeline component' = None,
         structure_model: 'Structure prediction approach' = 'heuristic',
         only_selected_models: 'Selection of models to evaluate' = None,
         fmt: 'Output format' = None,
         verbose=False):
    """Evaluate pipeline"""
    set_verbosity(verbose)

    def read_rows(name):
        fname = os.path.join(basedir, 'csv', name)
        try:
            return list(csv.reader(open(fname)))
        except:
            try:
                return list(csv.reader(open(fname, encoding='cp1252')))
            except:
                return []

    import collect, annotate, evaluate
    import structure, tabletypes, columns
    import dpath.util
    import pandas as pd
    import itertools

    table_metadata = list(annotate.get_metadata(basedir, annotationdir))
    items = [(meta, read_rows(meta['url'])) for meta in table_metadata]
    annotated_table_urls = set([meta['url'] for meta in table_metadata])

    pipeline = [
        ('Header Rows', ['struct',
                         'head'], structure, 'dialect/headerRowCount',
         evaluate.accuracy, evaluate.pipeline_accuracy),
        ('Transpose', ['struct',
                       'trans'], structure, 'dialect/karma:isTransposed',
         evaluate.precision_recall_f1, evaluate.pipeline_precision_recall_f1),
        ('Table Types', ['typ'], tabletypes, 'rdf:type', evaluate.accuracy,
         evaluate.pipeline_accuracy),
        ('Columns', ['col',
                     'colflat'], columns, 'tableSchema/columns/*/propertyUrl',
         evaluate.precision_recall_f1, evaluate.pipeline_precision_recall_f1),
        #         ('Columns Hierarchical', ['col', 'coltree'], columns, 'tableSchema/columns/*/propertyUrl',
        #          evaluate.h_precision_recall_f1, evaluate.pipeline_hierarchical_precision_recall_f1)
    ]

    if fmt == 'tex':

        def format_df(scores):
            return scores.to_latex()
    elif fmt == 'md':
        import tabulate

        def format_df(scores):
            return tabulate.tabulate(scores, tablefmt="pipe", headers="keys")
    else:

        def format_df(scores):
            return scores.to_string(line_width=None)

    for component_name, keywords, module, eval_var_path, eval_func, pipeline_scoring_func in pipeline:
        if (not component) or any(k in component for k in keywords):

            titleprefix = '#' if fmt == 'md' else ''
            print(titleprefix, component_name)

            scores = {}
            for model_name, (modelclass, kwargs) in module.models.items():
                if (not only_selected_models) or (model_name
                                                  in only_selected_models):
                    log.info('Evaluating %s...', model_name)
                    # Make model and evaluate
                    model = modelclass(modeldir, **kwargs)

                    # If the model has a pipeline, perform cross-validation with pipeline_scoring_func
                    if hasattr(model, 'pipeline'):

                        # If the kwargs of this parameterized model includes n_unlabeled, do self-training
                        unlabeled_items = []
                        n_unlabeled = kwargs.get('n_unlabeled', 0)
                        if n_unlabeled:
                            unlabeled_meta = []
                            for meta in collect.get_all_metadata(basedir):
                                if len(unlabeled_meta) >= n_unlabeled:
                                    break
                                if meta['url'] not in annotated_table_urls:
                                    unlabeled_meta.append(meta)
                            unlabeled_items = [(meta, read_rows(meta['url']))
                                               for meta in unlabeled_meta]

                            import tqdm
                            unlabeled_items = [
                                (struct_model.transform(dict(m), r), r)
                                for m, r in tqdm.tqdm(unlabeled_items,
                                                      'Structuring unlabeled')
                            ]

                        cv_scores = list(
                            evaluate.cross_validate(
                                model_name,
                                model,
                                items,
                                pipeline_scoring_func,
                                n_splits=5,
                                unlabeled_items=unlabeled_items,
                                verbose=verbose,
                                **kwargs))

                        def agg(x):
                            if x.name not in [
                                    'items', 'correct', 'tp', 'fp', 'fn'
                            ]:
                                if x.std() != 0:
                                    return '%.3f ± %.3f' % (x.mean(), x.std())
                                else:
                                    return int(x.mean())
                            else:
                                return int(x.sum())

                        cv_scores = pd.DataFrame.from_records(cv_scores).apply(
                            agg, axis=0)
                        scores[model_name] = dict(cv_scores)

                    else:
                        # If it does not have a pipeline, run eval_func
                        predicted = evaluate.yield_predicted_items(
                            model_name,
                            items,
                            model.transform,
                            eval_var_path,
                            verbose=verbose)
                        predicted = dict(i for p in predicted
                                         for i in p.items())
                        predicted = pd.DataFrame.from_dict(predicted,
                                                           orient='index')
                        try:
                            scores[model_name] = eval_func(
                                predicted['gold'], predicted['pred'])
                        except Exception as e:
                            print(predicted['gold'].isna().sum(),
                                  predicted['pred'].isna().sum())
                            raise e

            scores = pd.DataFrame.from_dict(scores, orient='index')
            print(format_df(scores))
            print()


@argh.arg('-s', '--selected-queries', nargs='*')
def label(
    modeldir: 'Model Directory',
    namespace: 'SPARQL namespace',
    kbdomain: 'SPARQL domain',
    selected_queries: 'Numeric indices of selected queries' = None,
    output_name: 'Output name (default: "results")' = None,
    parallel_threads=0,
    verbose=False,
):
    set_verbosity(verbose)

    if output_name:
        results_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                     output_name + '.json')
    else:
        results_fname = None

    import supervise
    supervise.cache_labelquery_results(modeldir,
                                       namespace,
                                       kbdomain,
                                       selected_queries=selected_queries,
                                       results_fname=results_fname,
                                       parallel=parallel_threads,
                                       verbose=verbose)


def analysis(
    basedir: 'Directory of PDF extractions',
    annotationdir: 'Annotation Directory',
    modeldir: 'Model Directory',
    namespace: 'SPARQL namespace',
    kbdomain: 'SPARQL domain',
    current_labelquery_type: '' = None,
    verbose=False,
):
    set_verbosity(verbose)

    results_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                 'gold.json')
    log.info(results_fname)
    if not os.path.exists(results_fname):
        log.info('Results file %s does not exist, rebuilding...',
                 results_fname)
        supervise.cache_labelquery_results(modeldir,
                                           namespace,
                                           kbdomain,
                                           results_fname=results_fname,
                                           verbose=verbose)

    labeled_metas = list(annotate.get_metadata(basedir, annotationdir))
    for labelquery_type in supervise.load_labelquery_results(
            modeldir, results_fname=results_fname):
        log.info('Analysing %s...', labelquery_type)
        if (not current_labelquery_type) or (labelquery_type
                                             == current_labelquery_type):
            supervise.save_query_analysis(modeldir, labeled_metas,
                                          labelquery_type)


import annotate


def labeltrain(
    basedir: 'Directory of PDF extractions',
    annotationdir: 'Annotation Directory',
    modeldir: 'Model Directory',
    task: 'Prediction task %s' % list(annotate.labelquery_types),
    output_picklefile: 'Output pickle file',
    structure_model: 'Structure prediction approach' = 'first row',
    type_model: 'Table type prediction approach' = 'supervised-lr',
    column_model: 'Column prediction approach' = 'supervised-lr',
    n_unlabeled: 'Number of unlabeled tables to use' = 1000,
    cache_name: 'Output name from cache' = 'results',
    exclude_queries: 'Queries to exclude' = "",
    maj: 'Use majority voting instead of label model' = False,
    subsample: 'Subsample large classes parameter' = 1.0,
    verbose=False,
):
    """Train model from label queries"""
    set_verbosity(verbose)

    subsample = float(subsample)

    if not os.path.exists(os.path.dirname(output_picklefile)):
        log.error('Output directory "%s" does not exist!',
                  os.path.dirname(output_picklefile))
        sys.exit(1)

    results_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                 cache_name + '.json')
    if not os.path.exists(results_fname):
        log.error(
            'Results file "%s" does not exist! Rebuild with command "label"',
            results_fname)
        sys.exit(1)
    log.info('Using results file "%s"', results_fname)

    log.info('Loading model...')
    import supervise
    task_labelqueries = supervise.load_labelquery_results(modeldir)
    if task not in task_labelqueries:
        log.error('Invalid task %s. Choose from %s', task,
                  list(task_labelqueries))
        sys.exit(1)
    if task == 'table-type':
        import tabletypes
        modelclass, kwargs = tabletypes.models[type_model]
        model_name = type_model
    if task == 'column-property':
        import columns
        modelclass, kwargs = columns.models[column_model]
        model_name = column_model
    model = modelclass(modeldir, **kwargs)
    labelqueries = task_labelqueries[task]
    if exclude_queries:
        exclude_queries = exclude_queries.split()
        log.info("Excluding queries %s", exclude_queries)
        labelqueries = {
            lqid: lq
            for lqid, lq in labelqueries.items() if lqid not in exclude_queries
        }

    def read_rows(name):
        fname = os.path.join(basedir, 'csv', name)
        try:
            return list(csv.reader(open(fname)))
        except:
            try:
                return list(csv.reader(open(fname, encoding='cp1252')))
            except:
                return []

    import annotate, collect, supervise
    log.info('Loading data...')
    iter_labeled_meta = annotate.get_metadata(basedir, annotationdir)
    iter_all_meta = collect.get_all_metadata(basedir)
    labeled_items, unlabeled_items = supervise.make_items(
        iter_labeled_meta, iter_all_meta, n_unlabeled, read_rows)
    if not unlabeled_items:
        unlabeled_items = list(labeled_items)
    log.info('Using %s labeled and %s unlabeled items', len(labeled_items),
             len(unlabeled_items))

    import tqdm, structure
    struct_modelclass, struct_kwargs = structure.models[structure_model]
    struct_model = struct_modelclass(modeldir, **struct_kwargs)
    structured_items = [
        (struct_model.transform(dict(m), r), r) for m, r in tqdm.tqdm(
            unlabeled_items, 'Structuring unlabeled', leave=verbose)
    ]

    log.info('Transform using label queries...')
    lX, L = supervise.transform_all(labelqueries, structured_items, model)
    L = L.replace({'UNK': pd.np.nan})
    eval_path = annotate.labelquery_types[task]['eval_path']
    log.info('Getting gold labels...')
    y_true = supervise.get_true_labelings(
        list(zip(*labeled_items))[0], eval_path)

    log.info('L shape is %s', L.shape)
    log.info('lX shape is %s', lX.shape)

    unk_value = annotate.labelquery_types[task]['UNK']
    L.loc[L.isna().min(axis=1), '-1'] = unk_value
    y_true = y_true.replace({None: unk_value}).fillna(unk_value)
    log.info('Added %d UNK values to y_true (total %s)',
             (y_true == unk_value).sum(), len(y_true))

    if maj:
        log.info('Getting label query majority voted predictions...')
        y_lpred = L.mode(axis=1)[0]

        log.info('Re-sampling class balance...')
        r = supervise.sample_match_dist(y_lpred, y_true,
                                        subsample=subsample).index

    else:
        log.info('Getting label model predictions...')
        y_lpred = supervise.labelmodel_predict(L, y_true, L)

        try:
            log.info('Subsampling with skew=%.2f', subsample)
            r = supervise.sample_match_dist(y_lpred,
                                            y_true,
                                            subsample=subsample).index
        except:
            log.info('Subsampling failed for skew=%.2f', subsample)
            r = y_lpred.index

    log.info(
        'Counts: \n %s',
        pd.DataFrame({
            'y_true': y_true.value_counts() / len(y_true),
            'y_lpred': y_lpred.loc[r].value_counts() / len(r),
        }))

    log.info('Training "%s" model...', model_name)
    model.pipeline.fit(lX.loc[r], y_lpred.loc[r])

    import pickle
    pickle.dump(model.pipeline, open(output_picklefile, 'wb'))


def modeleval(
    basedir: 'Directory of PDF extractions',
    annotationdir: 'Annotation Directory',
    table_model: 'Column model pkl file' = None,
    column_model: 'Table model pkl file' = None,
):
    import warnings
    warnings.filterwarnings("ignore")

    from query import get_table_test, get_column_test
    import pickle, pandas as pd

    if table_model:
        tb_model = pickle.load(open(table_model, 'rb'))

        X, y = get_table_test(basedir, annotationdir)
        import annotate
        task = 'table-type'
        unk_value = annotate.labelquery_types[task]['UNK']
        y = y.replace({None: unk_value, 'UNK': unk_value}).fillna(unk_value)

        print('\n{:>15} query-model accuracy: {acc:<10.3f}'.format(
            'Table', acc=tb_model.score(X, y)))

        print(pd.Series(tb_model.predict(X)).value_counts())

    if column_model:
        col_model = pickle.load(open(column_model, 'rb'))

        X, y = get_column_test(basedir, annotationdir)
        import annotate
        task = 'column-property'
        unk_value = annotate.labelquery_types[task]['UNK']
        y = y.replace({None: unk_value, 'UNK': unk_value}).fillna(unk_value)

        print('{:>15} query-model accuracy: {acc:<10.3f}'.format(
            'Column', acc=col_model.score(X, y)))

        print(pd.Series(col_model.predict(X)).value_counts())


if __name__ == '__main__':
    argh.dispatch_commands([run, eval, label, labeltrain, analysis, modeleval])
