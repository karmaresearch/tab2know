import rdflib, json, sys, dpath.util


def convert_predictions(metadata):
    table_node = metadata['@id']
    triples = []

    def triple(s, p, o):
        return ('<%s>' % s, '<%s>' % p, '<%s>' % o)

    rdf_type = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    if metadata['rdf:type']:
        triples.append(triple(table_node, rdf_type, metadata['rdf:type']))

    csvw_propertyUrl = 'http://www.w3.org/ns/csvw#propertyUrl'
    p = 'tableSchema/columns/*/propertyUrl'
    for path, val in dpath.util.search(metadata, p, yielded=True):
        _, _, colnr, _ = path.split('/')
        colnr = int(colnr) + 1
        if val:
            triples.append(
                triple(table_node + f'#col={colnr}', csvw_propertyUrl, val))

    return '\n'.join(' '.join(t) + ' .' for t in triples)


def convert(metadatastr, reader_func):
    g = rdflib.Graph().parse(data=metadatastr, format='json-ld')
    g.ns = {k: rdflib.Namespace(v) for k, v in g.namespaces()}

    csvw = g.ns['csvw']
    rdf = g.ns['rdf']
    dc = g.ns['dc']
    karma = rdflib.Namespace('http://karmaresearch.net/')

    for table_node, _, name_node in g.triples((None, csvw.url, None)):

        name = str(name_node)
        rows = reader_func(name)

        # https://www.w3.org/TR/tabular-metadata/#uri-template-properties
        table_ns = rdflib.Namespace(str(table_node))

        dialect = g.value(table_node, csvw.dialect)

        n_head = g.value(dialect, csvw.headerRowCount)

        if (n_head is not None):
            try:
                n_head = int(n_head)
            except ValueError:
                n_head = 0

        isTransposed = g.value(dialect, karma.isTransposed)
        if isTransposed:
            rows = list(zip(*rows))

        g.add((table_node, rdf.type, csvw.Table))
        g.add((table_node, karma.numberOfRows,
               rdflib.Literal(len(rows[n_head:]))))

        tableSchema = g.value(table_node, csvw.tableSchema)

        tableSchema_propertyUrl = g.value(
            tableSchema,
            csvw.propertyUrl) or '#col={_column}'  # log: add 'col='
        aboutUrl = g.value(tableSchema, csvw.aboutUrl) or '#row={_row}'

        try:
            columns = dict(
                enumerate(
                    rdflib.graph.Collection(g, g.value(tableSchema,
                                                       csvw.column))))
        except ValueError:
            # Possibly recursive, ignore table
            continue

####################################### Surface Graph ################################################
# Add column
        for ci, col in enumerate(zip(*rows)):
            col_node = rdflib.URIRef(table_ns + f"#col={ci+1}")

            g.add((table_node, csvw.column, col_node))
            g.add((col_node, rdf.type, csvw.Column))
            g.add((col_node, karma.colIndex,
                   rdflib.Literal(ci + 1)))  # add column colIndex

# Add column titles (headers created by heuristic model)
        for ci, column in columns.items():
            col_node = rdflib.URIRef(
                table_ns + f"#col={ci+1}"
            )

            for _, _, col_title in g.triples((column, csvw.title, None)):
                if col_title:
                    g.add((col_node, dc.title, rdflib.Literal(col_title)))

# Add row
        for ri, row in enumerate(rows):
            row_node = rdflib.URIRef(table_ns + f"#row={ri+1}")

            g.add((table_node, csvw.row, row_node))
            g.add((row_node, rdf.type, csvw.Row)) # different from csvw.rownum namespace added in the end (karma.rowIndex includes header rows)
            g.add((row_node, karma.rowIndex,
                   rdflib.Literal(ri + 1)))  # add row rowIndex

# Add cells
            for ci, cell in enumerate(row):
                if cell:
                    cell_node = rdflib.URIRef(table_ns +
                                              f"#row={ri+1}-col={ci+1}")

                    g.add((table_node, csvw.cell, cell_node))  # Cell type

                    g.add((cell_node, karma.rowIndex,
                           rdflib.Literal(ri + 1)))  # add cell rowIndex
                    g.add((cell_node, karma.colIndex,
                           rdflib.Literal(ci + 1)))  # add cell colIndex

                    g.add((cell_node, rdf.type, csvw.Cell))
                    g.add(
                        (cell_node, rdf.value, rdflib.Literal(
                            eval_value(cell))))  # add cell value with type

############################################## Semantic Graph #######################################################
        context = {}

        for ri, row in enumerate(rows[n_head:]):
            context['_row'] = ri + 1 + (
                n_head or 0)  ## adhere to added rows and cells above

            row_node = table_ns[aboutUrl.format(**context)]

            g.add(
                (row_node, csvw.rownum, rdflib.Literal(ri + 1))
            )  ## csvw.rownum still excludes headers for later semantic graph

            row_values = {}
            for ci, cell in enumerate(row):
                context['_column'] = ci + 1
                context['_name'] = None

                columnSchema = columns.get(ci)
                if columnSchema:
                    columnSchema_propertyUrl = g.value(
                        columnSchema,
                        csvw.propertyUrl) or tableSchema_propertyUrl
                    propertyName = g.value(columnSchema, csvw.name)

                    if propertyName:
                        row_values[propertyName] = cell

# TODO datatype: [ Done: eval_value() / get_cell_type() ]
                    propertyUrl = columnSchema_propertyUrl.format(**context)
                    if not propertyUrl.startswith('#'):
                        propertyNode = rdflib.URIRef(propertyUrl)

                        ## only include evaluated property values: <row,property,val>
                        g.add((row_node, propertyNode,
                               rdflib.Literal(eval_value(cell))))

                    else:
                        propertyNode = table_ns[propertyUrl]
    return g

def get_cell_type(cell: str) -> type:  # later
    import re
    cell = cell.strip().lower()
    num_digit = 0
    for char in cell:
        if char.isdigit():
            num_digit += 1
    if num_digit / len(cell) > .3:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", cell)
        for num in nums:
            if '.' in num:
                return float
            else:
                return int
    elif cell == 'true' or cell == 'false':
        return bool


def eval_value(cell):
    cell = cell.strip().lower()
    try:
        if '.' in cell:
            val = float(cell)
        else:
            val = int(cell)
    except ValueError:
        if cell == 'true' or cell == 'false':
            val = eval(cell.capitalize())
        else:
            val = cell
    return val


if __name__ == '__main__':

    def convert(in_fmt, out_fmt):
        return rdflib.Graph().parse(
            sys.stdin, format=in_fmt).serialize(format=out_fmt).decode('utf8')

    import argh
    argh.dispatch_commands([convert])
