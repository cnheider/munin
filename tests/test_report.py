#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from draugr.visualisation import confusion_matrix_plot, roc_plot
from matplotlib import pyplot
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sorcery import dict_of
from warg import NOD

from munin.html_embeddings import plt_html_svg, plt_html
from munin.plugins.dynamic.cf import generate_metric_table

pyplot.rcParams["figure.figsize"] = (3, 3)
import numpy
from pathlib import Path
from munin.generate_report import generate_pdf, generate_html, ReportEntry


def test_generation(do_generate_pdf=False):
    data_path = Path.home()
    num_classes = 3
    cell_width = int((800 / num_classes) - 6 - 6 * 2)

    pyplot.plot(numpy.random.random((3, 3)))

    a = ReportEntry(
        name=1,
        figure=plt_html_svg(size=(cell_width, cell_width)),
        prediction="a",
        truth="b",
        outcome="fp",
        explanation=None,
    )

    pyplot.plot(numpy.ones((9, 3)))

    b = ReportEntry(
        name=2,
        figure=plt_html(format="svg", size=(cell_width, cell_width)),
        prediction="b",
        truth="c",
        outcome="fp",
        explanation=None,
    )

    pyplot.plot(numpy.ones((5, 6)))

    c = ReportEntry(
        name=3,
        figure=plt_html(size=(cell_width, cell_width)),
        prediction="a",
        truth="a",
        outcome="tp",
        explanation=None,
    )

    d = ReportEntry(
        name="fas3",
        figure=plt_html(format="jpg", size=(cell_width, cell_width)),
        prediction="a",
        truth="a",
        outcome="tp",
        explanation=None,
    )

    e = ReportEntry(
        name="fas3",
        figure=plt_html(format="jpeg", size=(cell_width, cell_width)),
        prediction="c",
        truth="c",
        outcome="tn",
        explanation=plt_html(format="svg", size=(cell_width, cell_width)),
    )

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    bina = LabelBinarizer()
    y = bina.fit_transform(y)
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    classifier = OneVsRestClassifier(svm.SVC(kernel="linear", probability=True))
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    y_p_max = y_pred.argmax(axis=-1)
    y_t_max = y_test.argmax(axis=-1)

    confusion_matrix_plot(y_t_max, y_p_max, category_names=class_names)

    title = "Classification Report"
    confusion_matrix = plt_html(format="png", size=(800, 800))
    predictions = [[a, b, d], [a, c, d], [a, c, b], [c, b, e]]

    metrics = generate_metric_table(y_t_max, y_p_max, class_names)

    roc_plot(y_pred, y_test, n_classes)

    roc_figure = plt_html(format="png", size=(800, 800))

    bundle = NOD(dict_of(title, confusion_matrix, metrics, predictions, roc_figure))

    file_name = title.lower().replace(" ", "_")

    generate_html(file_name, **bundle)
    if do_generate_pdf:
        generate_pdf(file_name)


if __name__ == "__main__":
    test_generation(True)
