from flask import Flask, escape, request, render_template, g, redirect, url_for
import requests
import dpath
from tab2know import annotate

# Run Flask: env PYTHONPATH=tab2know/ FLASK_ENV=development FLASK_APP=app.py BASEDIR=data/extracted-data ANNOTATIONDIR=annotations/ MODELDIR=models/ QUERYDIR=test-queries/ flask run --port=5555
app = Flask(__name__)


def table_html(metadata, rows):
    import pandas as pd
    headerRowCount = dpath.util.get(
        metadata, 'dialect/headerRowCount', default=0) or 0
    isTransposed = dpath.util.get(
        metadata, 'dialect/karma:isTransposed', default=0) or 0
    if isTransposed:
        rows = list(zip(*rows))

    if rows[:headerRowCount]:
        ncols = len(rows[0])
        cols = [list(range(ncols))] + list(rows[:headerRowCount])
        cols = pd.MultiIndex.from_tuples(list(zip(*cols)))
    else:
        cols = None
    df = pd.DataFrame(rows[headerRowCount:], columns=cols)
    return df.to_html()


def do_sparql(kb_domain, query, namespace, **kwargs):
    params = {
        'format': 'json',
        'namespace': namespace,
        'query': query,
        **kwargs
    }
    return requests.get(kb_domain + '/sparql', params=params)


def try_image(name):
    basedir = os.environ.get("BASEDIR")
    if name:
        fimg = os.path.realpath(os.path.join(basedir, 'img', name))
        if fimg and os.path.exists(fimg) and not os.path.isdir(fimg):
            with open(fimg, 'rb') as fr:
                return base64.encodebytes(fr.read()).strip().decode('ascii')


def get_namespaces(kb_domain):
    ns = set()
    url = kb_domain + '/bigdata/namespace?format=json'
    for b in requests.get(url).json().get('results', {}).get('bindings', []):
        if b.get('predicate', {}).get('value',
                                      '') == 'http://purl.org/dc/terms/title':
            ns.add(b.get('object', {}).get('value', ''))
    return sorted(ns)


app.jinja_env.globals.update(try_image=try_image)


@app.route('/')
def home():
    querydir = os.environ.get("QUERYDIR")
    example_queries = {}
    for fname in os.listdir(querydir):
        name, ext = os.path.splitext(fname)
        if ext == '.sparql':
            example_queries[name] = open(os.path.join(querydir,
                                                      fname)).read().strip()

    return render_template(
        'home.html',
        example_queries=example_queries,
        namespace=request.cookies.get('namespace',
                                      os.environ.get("ANNOTATION_NAMESPACE")),
        all_namespaces=get_namespaces(os.environ.get("KB")),
    )


@app.route('/sparql', methods=['GET', 'POST'])
def sparql():
    params = dict(request.form)
    params['namespace'] = request.cookies.get(
        'namespace', os.environ.get("ANNOTATION_NAMESPACE"))
    params['format'] = 'json'
    return requests.get(os.environ.get("KB") + '/sparql', params=params).text


@app.route('/view')
def view():
    urls = request.args.getlist("url", None)
    label = request.args.get("label", None)
    lqid = request.args.get("lqid", None)
    showtype = request.args.get("showtype", None)
    evalset = request.args.get("evalset", 'POS')
    description = ''

    if label and not showtype:
        title = label
        for url, (meta, rows) in get_db().items():
            props = dpath.util.values(meta,
                                      'tableSchema/columns/*/propertyUrl')
            if (meta.get('rdf:type') == label) or (label in props):
                urls.append(url)

    true_values, pos_values = {}, {}
    if lqid or (label and showtype):
        modeldir = os.environ.get("MODELDIR")
        all_items, pos_items, true_items = set(), set(), set()

        if lqid:
            labelqueries, templates = load_labelqueries_templates(
                modeldir, filter_disabled=False)
            if lqid in labelqueries:
                lq = labelqueries[lqid]
                description = '{template}({slots})'.format(**lq)
                value = lq['value']

                eval_path = annotate.labelquery_types[lq['label']]['eval_path']
                all_values = {(url, p): v
                              for url, (meta, _) in get_db().items()
                              for p, v in dpath.util.search(
                                  meta, eval_path, yielded=True)}
                true_items = set(
                    [i for i, v in all_values.items() if value == v])

                lq = annotate.make_labelquery(
                    lq['label'],
                    lq['template'],
                    lq['slots'],
                    lq['value'],
                    templates,
                    os.environ.get("ANNOTATION_NAMESPACE"),
                    kbdomain=os.environ.get("KB"))
                pos_items = set((uri.split('/')[-1], t)
                                for uri, ts in lq.transformations.items()
                                for t in ts)
                all_items = set(all_values)
                pos_values = lq.transformations
        elif label and showtype:
            description = f'aggregated {showtype} queries'
            os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'),
                        exist_ok=True)
            labeing_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                         f'{showtype}-labeling.csv')
            if os.path.exists(labeing_fname):
                y = pd.read_csv(labeing_fname, index_col=[0, 1])
                all_items = set(y.index)
                pos_items = set(y.index[(y['pred'] == label)])
                true_items = set(y.index[(y['true'] == label)])
                all_values = dict(y['true'])
                pos_values = {
                    k: dict(v.droplevel(0))
                    for k, v in y['pred'].groupby(level=0)
                }
            else:
                description += ' [error: results cache file does not exist]'

        item_sets = {
            'POS': pos_items,
            'NEG': all_items - pos_items,
            'TP': pos_items & true_items,
            'FP': pos_items - true_items,
            'FN': true_items - pos_items,
            'TN': all_items - true_items - pos_items,
        }

        l = len(item_sets.get(evalset, []))
        if lqid:
            title = f'query {lqid}: {l} {evalset.upper()} items'
            description = f'Labeled "{value.split("/")[-1]}" using {description}'
        elif label and showtype:
            title = f'aggregated labeling for {label.split("/")[-1]}: {l} {evalset.upper()} items'
            description = f'Labeled "{label.split("/")[-1]}" using {description}'

        print(len(item_sets.get(evalset, [])), 'item sets')
        print(len(urls), 'urls')

        for url, p in item_sets.get(evalset, []):
            urls.append(url)
            true_values.setdefault(url, {})[p] = all_values.get((url, p))
            pos_values.setdefault(url, {})[p] = pos_values.get(url, {}).get(p)

    print(len(urls), 'urls')

    tables = []
    for url in sorted(set(urls)):
        metadata, rows = get_db().get(url, ({}, []))
        if metadata:
            tables.append(
                dict(
                    metadata=metadata,
                    rows=rows,
                    tablehtml=table_html(metadata, rows),
                    imgbase64=try_image(metadata.get("schema:image")),
                    true_values=true_values.get(url, {}),
                    pos_values=pos_values.get(url, {}),
                ))

    return render_template(
        'view_tables.html',
        label=label,
        title=title,
        description=description,
        lqid=lqid,
        showtype=showtype,
        evalset=evalset,
        urls=urls,
        tables=tables,
        namespace=request.cookies.get('namespace',
                                      os.environ.get("ANNOTATION_NAMESPACE")),
        all_namespaces=get_namespaces(os.environ.get("KB")),
    )


@app.route('/explore', methods=['GET', 'POST'])
def explore():
    uri = request.args.get("uri", None, type=str)
    namespace = request.cookies.get('namespace',
                                    os.environ.get("ANNOTATION_NAMESPACE"))
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'set-namespace':
            resp = redirect(request.referrer)
            val = request.form.get('namespace', default=None, type=str)
            if val is not None:
                resp.set_cookie('namespace', val)
            return resp

    pos = []
    po_query = "select ?p ?o where { <%s> ?p ?o.  } limit 100" % uri
    po_response = do_sparql(os.environ.get("KB"), po_query, namespace)
    if po_response.ok:
        pos = [
            (b['p'], b['o'])
            for b in po_response.json().get('results', {}).get('bindings', [])
        ]

    sps = []
    sp_query = "select ?s ?p where { ?s ?p <%s> .  } limit 100" % uri
    sp_response = do_sparql(os.environ.get("KB"), sp_query, namespace)
    if sp_response.ok:
        sps = [
            (b['s'], b['p'])
            for b in sp_response.json().get('results', {}).get('bindings', [])
        ]

    return render_template(
        'explore.html',
        uri=uri,
        pos=pos,
        sps=sps,
        namespace=namespace,
        all_namespaces=get_namespaces(os.environ.get("KB")),
    )


from annotate import *


def get_db():
    if 'db' not in g:
        basedir = os.environ.get("BASEDIR")
        annotationdir = os.environ.get("ANNOTATIONDIR")

        def read(name):
            fname = os.path.join(basedir, 'csv', name)
            return list(csv.reader(open(fname)))

        rebuild = False
        g.metadatafile = os.path.join(annotationdir, 'all_metadata.jsonl')
        if rebuild or (not os.path.exists(g.metadatafile)):
            docs = list(get_metadata(basedir, annotationdir))
            with open(g.metadatafile, 'w') as fw:
                for doc in docs:
                    print(json.dumps(doc), file=fw)

        metadatas = [json.loads(line) for line in open(g.metadatafile)]
        g.db = {m['url']: (m, read(m['url'])) for m in metadatas}

    return g.db


def get_classes():
    if 'classes' not in g:
        modeldir = os.environ.get("MODELDIR")
        import glob, rdflib
        gr = rdflib.Graph()
        if modeldir:
            for fname in glob.glob(os.path.join(modeldir, '*.ttl')):
                gr.parse(fname, format="n3")

        g.classes = set(
            gr.subjects(rdflib.namespace.RDF.type, rdflib.namespace.OWL.Class))
    return g.classes


@app.route('/set-namespace', methods=['POST'])
def set_namespace():
    resp = redirect(request.referrer)
    val = request.form.get('namespace', default=None, type=str)
    if val is not None:
        resp.set_cookie('namespace', val)
    return resp


@app.route('/annotate', methods=['GET', 'POST'])
def interface():
    url = request.args.get("url", None, type=str)

    global form_paths

    current_form_paths = list(form_paths)

    if url is None:
        for meta, _ in get_db().values():
            print('metadata keys:', *meta.keys())
            break
        table_path_annotations = {
            table: {
                p: [(str(v) if (v is not None) else '')
                    for _, v in dpath.util.search(meta, p, yielded=True)]
                for p in current_form_paths
            }
            for table, (meta, _) in get_db().items()
        }
        return render_template(
            'list_tables.html',
            db=get_db(),
            table_path_annotations=table_path_annotations,
            form_paths=current_form_paths,
            namespace=request.cookies.get(
                'namespace', os.environ.get("ANNOTATION_NAMESPACE")),
            all_namespaces=get_namespaces(os.environ.get("KB")),
        )
    else:
        metadata, rows = get_db().get(url, ({}, []))
        if metadata:
            urls = list(get_db())
            next_table = urls[min(urls.index(url) + 1, len(urls) - 1)]
            prev_table = urls[urls.index(url) - 1]

            # For non-glob form paths, add default None value
            for form_path in current_form_paths:
                if '*' not in form_path:
                    if dpath.util.get(metadata, form_path,
                                      default=None) is None:
                        dpath.util.new(metadata, form_path, None)

            # Make form
            if request.method == 'POST':
                newpth = request.form.get('newpth')
                newval = request.form.get('newval')
                if newpth and newval:
                    print(newpth, newval)
                    current_form_paths.append(newpth)
                    dpath.util.new(metadata, newpth, newval)

            form_elements = []
            for form_path in current_form_paths:
                for pth, obj in dpath.util.search(metadata,
                                                  form_path,
                                                  yielded=True):
                    if request.method == 'POST':
                        # Get new values
                        newval = request.form.get(pth)
                        if newval is not None:
                            typ = int if newval.isnumeric() else str
                            obj = typ(newval)
                            dpath.util.set(metadata, pth, obj)
                    form_elements.append({
                        'name':
                        pth,
                        'value':
                        '' if obj is None else obj,
                        'class':
                        'autocomplete' if form_path in [
                            'rdf:type', 'tableSchema/columns/*/propertyUrl'
                        ] else ''
                    })

            # Update
            if request.method == 'POST':
                g.db[url] = (metadata, rows)
                annotationdir = os.environ.get("ANNOTATIONDIR")
                with open(g.metadatafile, 'w') as fw:
                    for doc, _ in get_db().values():
                        print(json.dumps(doc), file=fw)

            # Make table HTML
            tablehtml = table_html(metadata, rows)

            # Get image
            imgbase64 = try_image(metadata.get("schema:image"))

            return render_template(
                'table.html',
                metadata=metadata,
                tablehtml=tablehtml,
                imgbase64=imgbase64,
                form_elements=form_elements,
                next_table=next_table,
                prev_table=prev_table,
                classes=sorted(get_classes()),
                namespace=request.cookies.get(
                    'namespace', os.environ.get("ANNOTATION_NAMESPACE")),
                all_namespaces=get_namespaces(os.environ.get("KB")),
            )
        else:
            return 'Table not in sample'


@app.route('/labelquery', methods=['GET', 'POST'])
def labelquery():
    query = "select ?s ?p ?o where { ?s ?p ?o. } limit 10"
    namespace = request.cookies.get('namespace',
                                    os.environ.get("ANNOTATION_NAMESPACE"))
    template = ''
    slots = ''
    results = {}

    current_labelquery_type = request.args.get("showtype",
                                               list(labelquery_types)[0],
                                               type=str)

    modeldir = os.environ.get("MODELDIR")
    labelqueries, templates = load_labelqueries_templates(
        modeldir, filter_disabled=False)

    if request.method == 'POST':
        form = dict(request.form)
        print(form)
        action = request.form.get('action')

        if action in [
                'query', 'count', 'setquery', 'save-template',
                'save-labelquery'
        ]:
            template = request.form.get('template')
            slots = request.form.get('slots')
            label = current_labelquery_type
            value = 'http://karmaresearch.net/' + request.form.get('value', '')
            query = request.form.get('query')
            if not query:
                query = templates[request.form.get('template')]

        if action in ['query', 'count']:
            fmt_query = fill_template(query, slots)
            print(fmt_query)
            response = do_sparql(os.environ.get("KB"), fmt_query, namespace)
            if response.ok:
                results = response.json()
                if action == 'count':
                    qid = request.form.get('qid')
                    var_path_template = annotate.labelquery_types[label][
                        'path']
                    n = len(
                        LabelQuery.transformations_from_results(
                            response.json(), var_path_template, value))
                    labelqueries[qid]['scores']['n_results'] = n
                    save_labelqueries_templates(modeldir, labelqueries,
                                                templates)
            else:
                print(response.text)

        if action == 'save-template':
            templates[template] = query

            # (re-)calculate all scores for labelqueries with this template
            basedir = os.environ.get("BASEDIR")
            annotationdir = os.environ.get("ANNOTATIONDIR")
            namespace = os.environ.get("ANNOTATION_NAMESPACE")
            for qid, lq in labelqueries.items():
                if lq['template'] == template:
                    scores = eval_labelquery(templates[lq['template']],
                                             lq['slots'],
                                             lq['label'],
                                             lq['value'],
                                             basedir,
                                             annotationdir,
                                             namespace,
                                             name='')
                    print(qid, scores)
                    labelqueries[qid]['scores'] = scores

        if action == 'build-cache':
            # (re-)calculate all scores for labelqueries of this labeltype
            basedir = os.environ.get("BASEDIR")
            annotationdir = os.environ.get("ANNOTATIONDIR")
            namespace = os.environ.get("ANNOTATION_NAMESPACE")
            for qid, lq in labelqueries.items():
                if lq['label'] == current_labelquery_type and lq[
                        'template'] in templates:
                    scores = eval_labelquery(templates[lq['template']],
                                             lq['slots'],
                                             lq['label'],
                                             lq['value'],
                                             basedir,
                                             annotationdir,
                                             namespace,
                                             name='')
                    print(qid, scores)
                    labelqueries[qid]['scores'] = scores

        if action == 'delete-labelquery':
            qid = request.form.get('qid')
            del labelqueries[qid]

        if action == 'save-labelquery':
            qid = request.form.get('qid')
            enabled = request.form.get('enabled')
            labelqueries[qid] = {
                'template': template,
                'slots': slots,
                'label': label,
                'value': value,
                'enabled': enabled,
            }
            if request.form.get('template') not in templates:
                templates[request.form.get('template')] = ''

            # (re-)calculate labelquery scores
            basedir = os.environ.get("BASEDIR")
            annotationdir = os.environ.get("ANNOTATIONDIR")
            namespace = os.environ.get("ANNOTATION_NAMESPACE")
            scores = eval_labelquery(templates[template],
                                     slots,
                                     label,
                                     value,
                                     basedir,
                                     annotationdir,
                                     namespace,
                                     name='')
            print('re-calculated', scores)
            labelqueries[qid]['scores'] = scores

        if action in [
                'save-template', 'delete-labelquery', 'save-labelquery',
                'build-cache'
        ]:
            # Save the queries themselves, including scores that have possibly been updated
            save_labelqueries_templates(modeldir, labelqueries, templates)

            # Run the updated queries on the gold namespace and save the results
            os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'),
                        exist_ok=True)
            results_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                         'gold.json')
            namespace = os.environ.get("ANNOTATION_NAMESPACE")
            kbdomain = os.environ.get("KB")
            if action == 'save-template':
                selected_queries = [
                    str(qid) for qid, lq in labelqueries.items()
                    if lq['template'] == template
                ]
            elif action == 'build-cache':
                selected_queries = [
                    str(qid) for qid, lq in labelqueries.items()
                    if lq['label'] == current_labelquery_type
                    and lq['template'] in templates
                ]
            else:
                selected_queries = [str(qid)]
            import supervise
            labelquery_results = supervise.cache_labelquery_results(
                modeldir,
                namespace,
                kbdomain,
                selected_queries=selected_queries,
                results_fname=results_fname,
                verbose=True)

            # Save the entire labeling matrix using the query results
            basedir = os.environ.get("BASEDIR")
            annotationdir = os.environ.get("ANNOTATIONDIR")
            labeled_metas = list(annotate.get_metadata(basedir, annotationdir))
            supervise.save_query_analysis(modeldir, labeled_metas,
                                          current_labelquery_type)

    view = request.args.get("view", None, type=str)
    if view:
        os.makedirs(os.path.join(modeldir, 'labelqueries', 'cache'),
                    exist_ok=True)
        view_fname = os.path.join(modeldir, 'labelqueries', 'cache',
                                  f'{current_labelquery_type}-{view}.csv')
        if os.path.exists(view_fname):
            pd.set_option('display.max_colwidth', -1)
            view = pd.read_csv(view_fname)
            view.columns = [
                c + '<a class="sort_btn" href="#"/>' for c in view.columns
            ]

            def make_url(c):
                c = 'http://karmaresearch.net/' + c if c != 'None' else c
                return url_for('view',
                               label=c,
                               showtype=current_labelquery_type)

            view[view.columns[0]] = [
                f'<a href="{make_url(c)}">{c}</a>'
                for c in view[view.columns[0]]
            ]
            view = view.to_html(index=False,
                                float_format='%.2f',
                                border=0,
                                escape=False,
                                classes=['sortable'])
        else:
            view = f'{view} file does not exist'

    return render_template(
        'labelquery.html',
        query=query,
        template=template,
        slots=slots,
        view=view,
        results=results,
        labelqueries=labelqueries,
        labelquery_types=labelquery_types,
        current_labelquery_type=current_labelquery_type,
        templates=templates,
        classes=sorted([
            c.replace('http://karmaresearch.net/', '') for c in get_classes()
        ]),
        new_qid=str(max([int(i) for i in labelqueries], default=0) + 1),
        namespace=request.cookies.get('namespace',
                                      os.environ.get("ANNOTATION_NAMESPACE")),
        all_namespaces=get_namespaces(os.environ.get("KB")),
    )
