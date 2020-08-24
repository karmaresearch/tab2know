import os, csv, collections, itertools
import fitz


def is_inside(inner, outer):
    return ((inner['x2'] >= outer['x1']) and (inner['y2'] >= outer['y1'])
            and (outer['x2'] >= inner['x1']) and (outer['y2'] >= inner['y1']))


def merge_words_hor(words, bound_x=1, bound_y=1):
    i_word = dict(enumerate(words))
    merge = {i: i for i in i_word}
    for i1, (bb1, w1) in i_word.items():
        for i2, (bb2, w2) in i_word.items():
            if i2 != i1:
                if abs(bb1['y1'] - bb2['y1']) < bound_y and abs(
                        bb1['x2'] - bb2['x1']) < bound_x:
                    merge[i2] = merge[i1]

    for i in set(merge.values()):
        merged = [i_word[k] for k, v in merge.items() if v == i]
        bb = dict(
            x1=min(bb['x1'] for bb, w in merged),
            y1=min(bb['y1'] for bb, w in merged),
            x2=max(bb['x2'] for bb, w in merged),
            y2=max(bb['y2'] for bb, w in merged),
        )
        bb['xc'] = (bb['x1'] + bb['x2']) / 2
        bb['yc'] = (bb['y1'] + bb['y2']) / 2
        w = ' '.join(w for _, w in merged)
        yield bb, w


def get_lines(words, rounding=1):
    def r(v):
        return int(v * rounding) / rounding

    point_bboxes = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for bbox, w in words:
        for k, v in bbox.items():
            point_bboxes[k][r(v)].append(bbox)
    for k, v_bboxes in point_bboxes.items():
        for v, bboxes in v_bboxes.items():
            v = sum(bb[k] for bb in bboxes) / len(bboxes)
            if len(bboxes) > 1:
                if k[0] == 'x':
                    points = dict(x1=v,
                                  y1=min(b['y1'] for b in bboxes),
                                  x2=v,
                                  y2=max(b['y2'] for b in bboxes))
                else:
                    points = dict(x1=min(b['x1'] for b in bboxes),
                                  y1=v,
                                  x2=max(b['x2'] for b in bboxes),
                                  y2=v)
                yield k, (v, points, bboxes)


def get_word_clusters(lines, words, bound_x=1, bound_y=1):
    def intersects(bb, p):
        bx, by = bound_x, bound_y
        bb = dict(x1=bb['x1'] - bx,
                  y1=bb['y1'] - by,
                  x2=bb['x2'] + bx,
                  y2=bb['y2'] + by)
        if p['x1'] == p['x2']:  # vertical
            return (p['x1'] >= bb['x1'] and p['x1'] <= bb['x2']
                    and p['y1'] <= bb['y2'] and p['y2'] >= bb['y1'])
        else:
            return (p['y1'] >= bb['y1'] and p['y1'] <= bb['y2']
                    and p['x1'] <= bb['x2'] and p['x2'] >= bb['x1'])

    for k, klines in itertools.groupby(sorted(lines), lambda x: x[0][0]):
        word_cluster = {i: i for i, _ in enumerate(words)}
        for _, (v, p, bboxes) in klines:
            intersecting = [
                i for i, (bb, w) in enumerate(words) if intersects(bb, p)
            ]
            if intersecting:
                for i in intersecting:
                    word_cluster[i] = word_cluster[intersecting[0]]

        cluster_word_pairs = sorted((v, k) for k, v in word_cluster.items())
        clusters = []
        for ci, wis in itertools.groupby(cluster_word_pairs, lambda x: x[0]):
            _, wis = zip(*wis)
            v = sum(bb[k + '1'] for i in wis
                    for bb, w in [words[i]]) / len(wis)
            clusters.append((v, wis))

        yield k, sorted(clusters)


def make_grid(pos_clusters, words):
    pos_wi_i = {}
    pos_n = {'x': 0, 'y': 0}
    for k, clusters in pos_clusters.items():
        pos_wi_i[k] = {}
        pos_n[k] = len(clusters)
        for ci, (v, wis) in enumerate(clusters):
            for wi in wis:
                pos_wi_i[k][wi] = ci

    grid = [['' for _ in range(pos_n['x'])] for _ in range(pos_n['y'])]
    if ('x' in pos_wi_i) and ('y' in pos_wi_i):
        for i, (bb, w) in enumerate(words):
            xi, yi = pos_wi_i['x'][i], pos_wi_i['y'][i]
            grid[yi][xi] = w if not grid[yi][xi] else grid[yi][xi] + ' ' + w
        return grid
    else:
        return None


import copy


def fill_singleton_rows(grid):
    grid = copy.deepcopy(grid)
    newrows = []
    for row in grid:
        vals = set(c for c in row[1:] if c)
        if len(vals) == 1:
            val = list(vals)[0]
            newrows += [[val for c in row]]
        else:
            newrows += [row]
    return newrows


def make_middle_spanning_cols(grid):
    grid = copy.deepcopy(grid)
    cols = [list(col) for col in zip(*grid)]
    newcols = []
    ci = 1
    while ci < len(cols):
        col = cols[ci]
        vals = set(c for c in col if c)
        if len(vals) == 1:
            val = list(vals)[0]
            ri = col.index(val)
            if (ci + 1 < len(cols)) and (not cols[ci - 1][ri]) and (
                    not cols[ci + 1][ri]):
                cols[ci - 1][ri] = cols[ci + 1][ri] = val
                newcols += [cols[ci - 1], cols[ci + 1]]
                ci += 3
                continue
        newcols += [cols[ci - 1]]
        ci += 1
    newcols += [cols[-1]]
    return list(zip(*newcols))


def prostprocess(grid):
    return fill_singleton_rows(make_middle_spanning_cols(grid))


def extract(pdf_fname, pagenr, fig_bbox, bound_x=4, bound_y=2, rounding=.5):
    doc = fitz.open(pdf_fname)
    page = doc[pagenr]

    words = []
    for x1, y1, x2, y2, word, _, _, _ in page.getTextWords():
        bbox = dict(x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    xc=(x1 + x2 / 2),
                    yc=(y1 + y2) / 2)
        if is_inside(bbox, fig_bbox):
            words.append((bbox, word))

    words = list(merge_words_hor(words, bound_x=bound_x, bound_y=bound_y))
    lines = list(get_lines(words, rounding=rounding))

    pos_clusters = dict(get_word_clusters(lines, words, bound_x=1, bound_y=0))
    grid = make_grid(pos_clusters, words)
    return prostprocess(grid)


def show_table(pdf_fname, pdffig):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    doc = fitz.open(pdf_fname)
    fig = pdffig

    pagenr = fig['page']
    page = doc[pagenr]
    fig_bbox = fig['regionBoundary']

    png_path = fig['renderURL']

    img = mpimg.imread(png_path)
    h, w = fig_bbox['x2'] - fig_bbox['x1'], fig_bbox['y2'] - fig_bbox['y1']
    plt.figure(figsize=(16, 16 * (h / w)))
    imgplot = plt.imshow(img)

    words = []
    for x1, y1, x2, y2, word, _, _, _ in page.getTextWords():
        bbox = dict(x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    xc=(x1 + x2 / 2),
                    yc=(y1 + y2) / 2)
        if any(t == word
               for t in fig['imageText']) and is_inside(bbox, fig_bbox):
            words.append((bbox, word))

    words = list(merge_words_hor(words, bound_x=4, bound_y=2))

    fx, fy = fig_bbox['x1'], fig_bbox['y1']
    dpi = fig['renderDpi']

    def render(x, y):
        return (x - fx) * dpi / 72, (y - fy) * dpi / 72

    ax = plt.axes()

    lines = list(get_lines(words, rounding=.5))

    for k, (v, p, bboxes) in lines:
        #             if k[0] == 'x':
        x1, y1 = render(p['x1'], p['y1'])
        x2, y2 = render(p['x2'], p['y2'])
        ax.plot([x1, x2], [y1, y2])


if __name__ == '__main__':
    import sys, argh, tqdm

    import logging as log

    def set_verbosity(v):
        if v:
            log.basicConfig(level=log.INFO)
        else:
            loglevel = getattr(log, os.environ.get('LOGLEVEL', ''),
                               log.WARNING)
            logfile = os.environ.get('LOGFILE', None)
            log.basicConfig(filename=logfile, level=loglevel)

    def extract_one(pdf_fname, pagenr, bbox_string, verbose=False):
        set_verbosity(verbose)

        x1, y1, x2, y2 = bbox_string.split(',')
        grid = extract(pdf_fname, int(pagenr),
                       dict(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)))
        cw = csv.writer(sys.stdout)
        for row in grid:
            cw.writerow(row)

    def extract_all(
        workdir: 'Working Directory containing extractions',
        pdfdir: 'PDF directory',
        outdir: 'Output Directory',
        verbose=False,
    ):
        set_verbosity(verbose)

        import glob, os
        os.makedirs(outdir, exist_ok=True)

        jsonfiles = glob.glob(os.path.join(workdir, 'json', '*'))
        log.info('There are %s json files', len(jsonfiles))
        pdf_json_pairs = []

        for jsonfile in tqdm.tqdm(jsonfiles,
                                  desc='Locating PDFs',
                                  leave=verbose):
            pdffile = os.path.join(
                pdfdir,
                os.path.splitext(os.path.basename(jsonfile))[0] + '.pdf')
            if os.path.exists(pdffile):
                pdf_json_pairs.append((pdffile, jsonfile))
        log.info('Processing %s pdf files', len(pdf_json_pairs))

        import json, traceback
        for pdffile, jsonfile in tqdm.tqdm(pdf_json_pairs,
                                           desc='Converting',
                                           leave=verbose):
            pdfname = os.path.splitext(os.path.basename(jsonfile))[0]
            figs = json.load(open(jsonfile))
            for fig in figs:
                try:
                    if fig.get('figType') == 'Table':
                        bbox = fig['regionBoundary']
                        pagenr = int(fig['page'])

                        grid = extract(pdffile, pagenr, bbox)

                        if grid:
                            imgurl = fig['renderURL']
                            outname = f"{pdfname}-Table{imgurl[:imgurl.rfind('-')].split('Table')[1]}.csv"
                            #                             outname = f"{pdfname}-Table{fig['name']}.csv"
                            fname = os.path.join(outdir, outname)
                            with open(fname, 'w') as fw:
                                cw = csv.writer(fw)
                                for row in grid:
                                    cw.writerow(row)

                except Exception as e:
                    log.exception(e)

    argh.dispatch_commands([extract_one, extract_all])
