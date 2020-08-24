import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def pd_report(y_true, y_pred, target_names=None, **kwargs):
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore",
                            category=UndefinedMetricWarning,
                            module="sklearn")

    y_true = pd.Series(y_true).fillna('UNK')
    y_pred = pd.Series(y_pred).fillna('UNK')
    if not target_names:
        target_names = sorted(set(y_true))
    if ('UNK' not in target_names) and ('UNK' in set(y_true.values)
                                        | set(y_pred.values)):
        target_names = ['UNK'] + target_names

    report = classification_report(y_true,
                                   y_pred,
                                   output_dict=True,
                                   target_names=target_names,
                                   **kwargs)
    df = pd.DataFrame(report).T
    df['support'] = df['support'].map(int)
    return df


def format_vertical_headers(df):
    # https://stackoverflow.com/a/56750930
    """Display a dataframe with vertical column headers"""
    styles = [
        dict(selector="th", props=[('width', '40px')]),
        dict(selector="tr:last-child th.col_heading",
             props=[("writing-mode", "vertical-rl"),
                    ('transform', 'rotateZ(180deg)'),
                    ('vertical-align', 'top')])
    ]
    return df.style.set_table_styles(styles)


def pd_confusion(y_true, y_pred, target_names, stack=False, **kwargs):

    y_true = pd.Series(y_true).fillna('UNK')
    y_pred = pd.Series(y_pred).fillna('UNK')
    if ('UNK' not in target_names) and ('UNK' in set(y_true.values)
                                        | set(y_pred.values)):
        target_names = ['UNK'] + target_names

    conf = confusion_matrix(y_true=y_true, y_pred=y_pred, **kwargs)
    pred = pd.MultiIndex.from_tuples([('Predicted', t) for t in target_names])
    true = pd.MultiIndex.from_tuples([('True', t) for t in target_names])
    conf = pd.DataFrame(conf, columns=pred, index=true)

    if stack:
        conf = conf.stack().rename(columns={'Predicted': 'count'})
        conf.index = conf.index.droplevel(0).rename(['true', 'predicted'])
        conf = conf['count'].sort_values(ascending=False)
        conf = conf[conf > 0].to_frame()
    else:
        m = conf.max().max()

        def blue_shade(x):
            v = 1 - ((x / m) / 2)
            r, g, b = (pd.np.array([v, v, 1]) * 255).astype(int)
            return 'background-color: #{:02x}{:02x}{:02x}'.format(r, g, b)

        conf = format_vertical_headers(conf).applymap(blue_shade)

    return conf


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from itertools import cycle


def precision_recall_plot(y_test, y_score, target_names):
    from sklearn.preprocessing import label_binarize
    # Use label_binarize to be multi-label like settings
    y_test = y_test.fillna('UNK')
    y_test = label_binarize(y_test, classes=sorted(set(y_test)))

    n_classes = len(target_names)
    precision, recall = {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test[:, i], y_score[:, i])

    colors = cycle(plt.cm.Dark2.colors)
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    labels += list(target_names)
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines,
               labels,
               loc='center left',
               bbox_to_anchor=(1, 0.5),
               prop=dict(size=14))

    return plt


from sklearn.metrics import roc_auc_score


def pd_auroc(y_true, y_score):
    from sklearn.preprocessing import label_binarize
    # Use label_binarize to be multi-label like settings
    classes = sorted(set(y_true))
    y_test = label_binarize(y_true, classes=classes)

    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore",
                            category=UndefinedMetricWarning,
                            module="sklearn")

    per_class = list(roc_auc_score(y_test, y_score, average=None))
    averages = ['macro', 'weighted']
    per_avg = [roc_auc_score(y_test, y_score, average=a) for a in averages]
    return pd.Series(per_class + per_avg,
                     index=classes + averages,
                     name='auroc')
