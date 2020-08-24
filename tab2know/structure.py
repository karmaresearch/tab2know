import pandas as pd, warnings
import json, dpath.util


def outlier_scores(feats):
    # Number of standard deviations that values in column are away from mean
    m, s = feats.mean(axis=0), feats.std(axis=0)
    s[s == 0] = pd.np.nan
    return abs(feats - m) / s


all_char_feats = {
    'chars':
    lambda s: sum(1 for c in s if True),
    'upper':
    lambda s: sum(1 for c in s if c.isupper()) / len(s) if s else 0,
    'lower':
    lambda s: sum(1 for c in s if c.islower()) / len(s) if s else 0,
    'space':
    lambda s: sum(1 for c in s if c.isspace()) / len(s) if s else 0,
    'num':
    lambda s: sum(1 for c in s if c.isdigit()) / len(s) if s else 0,
    'not-alnum':
    lambda s: sum(1 for c in s if not c.isalnum()) / len(s) if s else 0,
}


def get_outliers(feats):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Get number of standard deviations away from mean for all features
        outliers = outlier_scores(feats)

        # Take the feature-averaged non-zero outlier scores
        outliers = pd.DataFrame(pd.np.nanmean(outliers, axis=-1))

    return outliers


def get_nhead(outliers, theta=1, verbose=False):
    # Get the index of the first row whose column-averaged outlier-score is not > 1
    nhead = 0
    if (~outliers.isna()).any().any():
        means = pd.np.nanmean(outliers, axis=1)
        if verbose:
            print('Outlier row scores:',
                  '[%s]' % ' '.join('%.2f' % r for r in means),
                  f'(shape={outliers.shape})')
        scores = (means > theta)
        nhead = list(scores).index(False) if not scores.all() else 0
    return nhead


def get_transpose_outliers(feats, max_rows=5, theta=1, verbose=False):
    feats_t = pd.np.transpose(feats, axes=(1, 0, 2))
    outliers_n = get_outliers(feats)
    outliers_t = get_outliers(feats_t)

    nhead_n = max(
        1, get_nhead(outliers_n.head(max_rows), theta=theta, verbose=verbose))
    nhead_t = max(
        1, get_nhead(outliers_t.head(max_rows), theta=theta, verbose=verbose))

    score_n = pd.np.nanmean(outliers_n.values[nhead_n:, :], axis=(0, 1))
    score_t = pd.np.nanmean(outliers_t.values[nhead_t:, :], axis=(0, 1))

    if verbose:
        print(f'transpose headers: normal={nhead_n} transpose={nhead_t:}')
        print(
            f'transpose scores: normal={score_n:.2f} transpose={score_t:.2f}')

    if score_t < (score_n * .85):
        return (True, outliers_t)
    else:
        return (False, outliers_n)


def heuristic_get_nhead_t(rows,
                          max_rows=5,
                          theta=1,
                          verbose=False,
                          char_feats=None):
    # Get features of cells
    df = pd.DataFrame(rows)
    char_feats = {c: all_char_feats[c]
                  for c in char_feats} if char_feats else all_char_feats
    feats = pd.np.dstack(
        [df.applymap(func) for name, func in all_char_feats.items()])

    h, w = df.shape
    if h > 2 and w > 2:
        transposed, outliers = get_transpose_outliers(feats,
                                                      max_rows=max_rows,
                                                      theta=theta,
                                                      verbose=verbose)
        if transposed:
            df = df.T
    else:
        transposed, outliers = False, get_outliers(feats)

    if verbose:
        print('Outlier matrix:')
        print(outliers)

    nhead = get_nhead(outliers.head(max_rows), theta=theta, verbose=verbose)
    return transposed, nhead


class FirstHeaderRow:
    def __init__(self, modeldir):
        pass


# add needed properties

    def transform(self, meta, rows, **kwargs):

        dpath.util.new(meta, 'dialect/headerRowCount', 1)
        dpath.util.new(meta, 'dialect/karma:isTransposed', 0)

        for ci, coltitles in enumerate(zip(*rows[:1])):
            dpath.util.new(meta, f'tableSchema/columns/{ci}/titles', coltitles)
        return meta


class Heuristic:
    def __init__(self, modeldir, theta=1, char_feats=None):
        self.theta = theta
        self.char_feats = char_feats

    def transform(self, meta, rows, verbose=False, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            transposed, nhead = heuristic_get_nhead_t(
                rows,
                theta=self.theta,
                verbose=verbose,
                char_feats=self.char_feats)

        dpath.util.new(meta, 'dialect/headerRowCount', nhead)
        dpath.util.new(meta, 'dialect/karma:isTransposed', int(transposed))

        for ci, coltitles in enumerate(zip(*rows[:nhead])):
            dpath.util.new(meta, f'tableSchema/columns/{ci}/titles', coltitles)
        return meta


models = {
    'first row': (FirstHeaderRow, {}),
    'heuristic': (Heuristic, {}),
    'heuristic (theta = 0.9)': (Heuristic, dict(theta=0.9)),
    'heuristic (theta = 0.8)': (Heuristic, dict(theta=0.8)),
}
# for n,f in all_char_feats.items():
#     models[f'heuristic (feat = {n})'] = (Heuristic, dict(char_feats=[n]))
