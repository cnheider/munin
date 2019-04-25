#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import BytesIO, StringIO

import markdown

__author__ = "cnheider"
__doc__ = ""

import matplotlib.pyplot as plt


def generate_math_html(equation="e^x", inline=True):
    """

  For inline math, use \(...\).
  For standalone math, use $$...$$, \[...\] or \begin...\end.
  md = markdown.Markdown(extensions=['mdx_math'])
  md.convert('$$e^x$$')

  :param equation:
  :param inline:
  :return:
  """
    md = markdown.Markdown(extensions=["mdx_math"], extension_configs={"mdx_math": {"add_preview": True}})
    if inline:
        stripped = md.convert(f"\({equation}\)").lstrip("<p>").rstrip("</p>")
        return f"<{stripped}"
    return md.convert(f"$${equation}$$")


def generate_qr():
    import pyqrcode
    import io
    import base64

    code = pyqrcode.create("hello")
    stream = io.BytesIO()
    code.png(stream, scale=6)
    png_encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return png_encoded


def plt_html_svg(*, size=(400, 400)):
    fig_file = StringIO()
    plt.savefig(fig_file, format="svg", dpi=100)
    fig_svg = f'<svg width="{size[0]}" height="{size[1]}" {fig_file.getvalue().split("<svg")[1]}'
    return fig_svg


def plt_html(title="image", *, format="png", size=(400, 400)):
    if format == "svg":
        return plt_html_svg(size=size)

    import base64

    fig_file = BytesIO()
    plt.savefig(fig_file, format=format, dpi=100)
    fig_file.seek(0)  # rewind to beginning of file
    fig_png = base64.b64encode(fig_file.getvalue()).decode("ascii")
    return f'<img width="{size[0]}" src="data:image/{format};base64,{fig_png}" alt="{title}" /><br>'


def plot_cf(y_pred, y_test, class_names, size=(8, 8), decimals=3):
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
        if not title:
            if normalize:
                title = "Normalized confusion matrix"
            else:
                title = "Confusion matrix, without normalization"

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        return ax

    np.set_printoptions(precision=decimals)

    # Plot normalized confusion matrix
    plot_confusion_matrix(
        y_test, y_pred, classes=class_names, normalize=True, title="Normalized confusion matrix"
    )
