import os, sys, csv, json

BASE = "http://karmaresearch.net/"


def get_pdffigures_meta(fname):
    tableid = os.path.basename(fname).replace('.csv', '')
    try:
        name, tablenr = tableid.rsplit('_table', 1)
    except ValueError:
        try:
            name, tablenr = tableid.rsplit('-Table', 1)
        except ValueError:
            name, tablenr = tableid, 1

    jsondir = os.path.join(os.path.dirname(os.path.dirname(fname)), 'json')
    jsonpath = os.path.join(jsondir, f'{name}.json')
    if os.path.exists(jsonpath):
        paper_figs = json.load(open(jsonpath))
        paper_tables = {
            fig['name']: fig
            for fig in paper_figs if fig['figType'] == 'Table'
        }
    else:
        paper_tables = {}

    pdfmeta = paper_tables.get(tablenr, {})
    pdfmeta['paper_id'] = name or ''
    pdfmeta['table_number'] = tablenr
    return pdfmeta


def get_file_meta(real_fname, paper_prefix='', id_lookup={}):
    pdfmeta = get_pdffigures_meta(real_fname)
    name = os.path.basename(real_fname)
    tableURI = BASE + "table/" + name

    rows = list(csv.reader(open(real_fname)))
    cols = list(zip(*rows))
    paper_id = id_lookup.get(pdfmeta['paper_id'], pdfmeta['paper_id'])

    return {
        "@id":
        tableURI,
        "@context": [
            "http://www.w3.org/ns/csvw", {
                'karma': BASE,
                'source': {
                    '@id': 'dc:source',
                    '@type': '@id'
                }
            }
        ],
        "url":
        name,
        "source": (paper_prefix + paper_id) if paper_id else None,
        "dc:title":
        pdfmeta.get('caption', None),
        "schema:pagination":
        pdfmeta.get('page', None),
        "schema:position":
        pdfmeta.get('table_number', None),
        "schema:image":
        os.path.basename(pdfmeta.get('renderURL', '')),
        "dialect": {
            "headerRowCount": None
        },
        "rdf:type":
        None,
        "tableSchema": {
            "columns": [{
                "propertyUrl": None,
            } for ci, c in enumerate(cols)]
        }
    }


def get_all_metadata(basedir, paper_prefix=''):
    f = os.path.join(basedir, 'id_lookup.txt')

    if not os.path.isfile(f):
        id_lookup = {}
    else:
        id_lookup = {
            a: b
            for line in open(f) for a, b in [line.strip().split()]
        }

    for name in os.listdir(os.path.join(basedir, 'csv')):
        real_fname = os.path.realpath(os.path.join(basedir, 'csv', name))
        if os.path.isfile(real_fname):
            yield get_file_meta(real_fname,
                                paper_prefix=paper_prefix,
                                id_lookup=id_lookup)
