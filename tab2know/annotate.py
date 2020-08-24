import os, sys, csv, json, base64, copy, logging as log, time
import pandas as pd, dpath.util
import collect

form_paths = [
    'dialect/headerRowCount',
    'dialect/karma:isTransposed',
    'dialect/karma:isMatrix',
    'rdf:type',
    'tableSchema/columns/*/propertyUrl',
    'karma:isBad',
    'karma:isHardForTabula',
]

labelquery_types = {
    'table-type': {
        'default-query': 'select ?table where {  }',
        'path': 'rdf:type',
        'eval_path': 'rdf:type',
        'UNK': 'http://karmaresearch.net/Other',
    },
    'column-property': {
        'default-query': 'select ?table ?colnr where {  }',
        'path': 'tableSchema/columns/{colnr}/propertyUrl',
        'eval_path': 'tableSchema/columns/*/propertyUrl',
        'UNK': 'http://karmaresearch.net/Other',
    },
}


def get_annotations(basedir, annotationdir):

    # No annotations, just selection
    fname = os.path.join(annotationdir, 'names.txt')
    if os.path.exists(fname):
        for line in open(fname):
            yield (line.strip(), 'url', line.strip())

    # Header rows
    fname = os.path.join(annotationdir, 'sample-annotations.tsv')
    if os.path.exists(fname):
        df = pd.read_csv(fname, sep='\t', index_col=None)
        for tup in df.itertuples():
            row = tup._asdict()
            yield (os.path.basename(row['fname']), 'dialect/headerRowCount',
                   row['number_of_headers'])

    # Table types


#     fname = os.path.join(annotationdir, 'type-annotations.tsv')
    fname = os.path.join(annotationdir, '200-annotations-new.tsv')
    if os.path.exists(fname):
        df = pd.read_csv(fname, sep='\t', index_col=None)
        #     id_type = ['ExperimentalResultsTable', 'MethodsTable', 'DatasetsTable', 'ConfigurationsTable', 'OtherTable']
        id_type = [
            'ExperimentalResultsTable', 'MethodComparisonTable',
            'ConfigurationOrFeatureTable', 'OtherTable'
        ]
        for row in df.to_dict('rows'):
            if row['Table Type'] is not None:
                yield (os.path.basename(row['Source']), 'rdf:type',
                       'http://karmaresearch.net/%s' %
                       id_type[int(float(row['Table Type']))])

    # Column properties
    fname = os.path.join(annotationdir, 'column-annotations.tsv')
    if os.path.exists(fname):
        df = pd.read_csv(fname, sep='\t', index_col=None)
        df = df[~df["Column Number"].isna()]
        for row in df.to_dict('rows'):
            if row['URI']:
                col = int(row["Column Number"])
                yield (os.path.basename(row['Paper ID']),
                       f'tableSchema/columns/{col}/propertyUrl',
                       row['URI'].strip())


def get_metadata(basedir, annotationdir, with_bad=False):
    def read(name):
        fname = os.path.join(basedir, 'csv', name)
        try:
            return list(csv.reader(open(fname)))
        except:
            try:
                return list(csv.reader(open(fname, encoding='cp1252')))
            except:
                return []

    metadatafile = os.path.join(annotationdir, 'all_metadata.jsonl')
    if os.path.exists(metadatafile):
        metadata = [
            json.loads(line) for line in open(metadatafile).readlines()
        ]

        # Make column headers
        for meta in metadata:
            nhead = meta.get('dialect', {}).get('headerRowCount')
            if nhead is not None:
                rows = read(meta['url'])
                for ci, coltitles in enumerate(zip(*rows[:1])):
                    try:
                        dpath.util.new(meta,
                                       f'tableSchema/columns/{ci}/titles',
                                       coltitles)
                    except Exception as e:
                        log.exception(e)

        for meta in metadata:
            # Get rid of bad tables
            if (not with_bad) and meta.get('karma:isBad', False):
                continue
            yield meta
    else:

        def get_meta(name):
            real_fname = os.path.realpath(os.path.join(basedir, 'csv', name))
            return collect.get_file_meta(real_fname)

        table_metadata = {}
        for name, pth, val in get_annotations(basedir, annotationdir):
            meta = table_metadata.setdefault(name, get_meta(name))
            dpath.util.new(meta, pth, val)

        for meta in table_metadata.values():
            # Get rid of bad tables
            if (not with_bad) and meta.get('karma:isBad', False):
                continue
            yield meta


def fill_template(template, slots):
    slots = dict(
        [s.strip().split('=', 1) for s in slots.split(',') if '=' in s])
    # switch single and double curly brackets (to match jinja syntax)
    template = template.replace('{{', '%<%').replace('}}', '%>%')
    template = template.replace('{', '{{').replace('}', '}}')
    template = template.replace('%<%', '{').replace('%>%', '}')
    return template.format(**slots)


import requests


def do_sparql(kb_domain, query, namespace, **kwargs):
    params = {
        'format': 'json',
        'namespace': namespace,
        'query': query,
        **kwargs
    }
    return requests.get(kb_domain + '/sparql', params=params)


class LabelQuery(object):
    def __init__(self,
                 transformations,
                 namespace,
                 kbdomain=os.environ.get("KB"),
                 name=None):
        self.transformations = transformations
        self.kbdomain = kbdomain
        self.namespace = namespace
        self.name = name

    @classmethod
    def from_query(cls,
                   query,
                   var_path_template,
                   value,
                   namespace,
                   kbdomain=os.environ.get("KB"),
                   name=None):
        kbdomain = kbdomain
        namespace = namespace
        transformations = {}
        # do query and get results in `response`
        response = do_sparql(kbdomain, query, namespace)
        if response.ok:
            transformations = cls.transformations_from_results(
                response.json(), var_path_template, value)
        return cls(transformations, namespace, kbdomain=kbdomain, name=name)

    @staticmethod
    def transformations_from_results(query_results, var_path_template, value):
        transformations = {}
        # make table -> variables map
        for r in query_results.get('results', {}).get('bindings', []):
            table = r.get('table', {}).get('value')
            if table:
                try:
                    var_path = var_path_template.format(
                        **{k: str(v.get('value', ''))
                           for k, v in r.items()})
                    transformations.setdefault(table, {})[var_path] = value
                except KeyError:
                    continue

        return transformations


    def transform(self, meta, rows, verbose=False, **kwargs):
        meta = copy.deepcopy(meta)
        for var_path, value in self.transformations.get(meta['@id'],
                                                        {}).items():
            try:
                dpath.util.new(meta, var_path, value)
            except Exception as e:
                if verbose:
                    print(e)
                    print('Cannot transform %s:' % meta['url'], var_path, '=',
                          value)
                    print(meta)
        return meta


def make_labelquery(querytype,
                    template,
                    slots,
                    value,
                    templates,
                    namespace,
                    kbdomain=os.environ.get("KB"),
                    name=None):
    global labelquery_types
    queryrep = f'{querytype} {name} [{template}({slots}) -> {value}]'
    now = time.time()
    log.info(f'Running {queryrep}')
    querytype = labelquery_types[querytype]
    template = templates[template]
    query = fill_template(template, slots)
    lq = LabelQuery.from_query(query,
                               querytype['path'],
                               value,
                               namespace,
                               kbdomain=kbdomain,
                               name=name)
    t = time.time() - now
    log.info(
        f'Found {len(lq.transformations)} results of {queryrep} in {t:.1f}s')
    return lq


import evaluate


def eval_labelquery(template,
                    slots,
                    label,
                    value,
                    basedir,
                    annotationdir,
                    namespace,
                    name=''):
    # make label query object
    query = fill_template(template, slots)
    var_path_template = labelquery_types[label]['path']
    lq = LabelQuery.from_query(query,
                               var_path_template,
                               value,
                               namespace=namespace)

    var_path = labelquery_types[label]['eval_path']
    table_metadata = get_metadata(basedir, annotationdir)
    gold_items = [(meta, []) for meta in table_metadata]

    predicted = evaluate.yield_predicted_items(name,
                                               gold_items,
                                               lq.transform,
                                               var_path,
                                               verbose=False)
    predicted = dict(i for p in predicted for i in p.items())
    predicted = pd.DataFrame.from_dict(predicted, orient='index')
    predicted = predicted[(predicted['gold'] == value) |
                          (predicted['pred'] == value)]

    scores = evaluate.precision_recall_f1(predicted['gold'], predicted['pred'])
    scores['matches'] = len(lq.transformations)
    return scores


def save_labelqueries_templates(modeldir, labelqueries, templates):
    templatedir = os.path.join(modeldir, 'labelqueries', 'templates')
    os.makedirs(templatedir, exist_ok=True)
    for name, template in templates.items():
        with open(os.path.join(templatedir, name + '.sparql'), 'w') as fw:
            print(template.strip(), file=fw)

    # split off scores without modifying referenced labelqueries
    labelqueries = {l: dict(lq) for l, lq in labelqueries.items()}
    scores = {k: dict(l.pop('scores', {})) for k, l in labelqueries.items()}
    resultstats = {k: v.pop('n_results', None) for k, v in scores.items()}

    os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'), exist_ok=True)
    scores_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                'scores.csv')
    with open(scores_fname, 'w') as fw:
        pd.DataFrame.from_dict(scores, orient='index').to_csv(fw)

    resultstats_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                     'results.stats.json')
    with open(resultstats_fname, 'w') as fw:
        json.dump({str(k): v for k, v in resultstats.items()}, fw)

    with open(os.path.join(modeldir, 'labelqueries', 'labelqueries.csv'),
              'w') as fw:
        pd.DataFrame.from_dict(labelqueries, orient='index').to_csv(fw)


def load_labelqueries_templates(modeldir, filter_disabled=True):
    labelqueries, templates = {}, {}

    templates_dir = os.path.join(modeldir, 'labelqueries', 'templates')
    for fname in os.listdir(templates_dir):
        name, ext = os.path.splitext(fname)
        if ext == '.sparql':
            templates[name] = open(os.path.join(templates_dir, fname)).read()

    labelqueries_fname = os.path.join(modeldir, 'labelqueries',
                                      'labelqueries.csv')
    df = pd.read_csv(labelqueries_fname, index_col=0).fillna('')
    labelqueries = {str(k): v for k, v in df.to_dict(orient='index').items()}

    scores = {}
    os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'), exist_ok=True)
    scores_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                'scores.csv')
    if os.path.exists(scores_fname):
        scores = pd.read_csv(scores_fname, index_col=0, converters={
            0: str
        }).to_dict(orient='index')

    resultstats = {}
    resultstats_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                     'results.stats.json')
    if os.path.exists(resultstats_fname):
        resultstats = json.load(open(resultstats_fname))

    for k in list(labelqueries):
        labelqueries[str(k)]['scores'] = scores.get(int(k), {})
        labelqueries[str(k)]['scores']['n_results'] = resultstats.get(
            str(k), None)
        if filter_disabled and (not labelqueries[str(k)].get('enabled')):
            del labelqueries[str(k)]

    return labelqueries, templates
