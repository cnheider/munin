#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from pathlib import Path

import numpy
from apppath import ensure_existence
from draugr.visualisation import confusion_matrix_plot, roc_plot
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sorcery import dict_of

from munin.generate_report import ReportEntry, generate_html, generate_pdf
from munin.html_embeddings import MetricEntry, plt_html, plt_html_svg
from munin.report_format import ReportFormatEnum
from munin.plugins.dynamic.cf import generate_metric_table


def a(
    title: str = "Classification Report", out_path=Path.cwd() / "exclude", num_classes=3, do_generate_pdf=True
):
    """description"""
    from matplotlib import pyplot

    pyplot.rcParams["figure.figsize"] = (3, 3)
    from warg.data_structures.named_ordered_dictionary import NOD

    ensure_existence(out_path)

    file_name = out_path / title.lower().replace(" ", "_")

    cell_width = int((800 / num_classes) - 6 - 6 * 2)

    pyplot.plot(numpy.random.random((3, 3)))

    GPU_STATS = ReportEntry(
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
        figure=plt_html(report_format=ReportFormatEnum.svg, size=(cell_width, cell_width)),
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
        figure=plt_html(report_format=ReportFormatEnum.jpg, size=(cell_width, cell_width)),
        prediction="a",
        truth="a",
        outcome="tp",
        explanation=None,
    )

    e = ReportEntry(
        name="fas3",
        figure=plt_html(report_format=ReportFormatEnum.jpeg, size=(cell_width, cell_width)),
        prediction="c",
        truth="c",
        outcome="tn",
        explanation=plt_html(report_format=ReportFormatEnum.svg, size=(cell_width, cell_width)),
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

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2)

    classifier = OneVsRestClassifier(svm.SVC(kernel="linear", probability=True))
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    y_p_max = y_pred.argmax(axis=-1)
    y_t_max = y_test.argmax(axis=-1)

    confusion_matrix = plt_html(
        confusion_matrix_plot(y_t_max, y_p_max, category_names=class_names),
        report_format=ReportFormatEnum.png,
        size=(800, 800),
    )
    predictions = [
        [GPU_STATS, b, d],
        [GPU_STATS, c, d],
        [GPU_STATS, c, b],
        [c, b, e],
    ]

    metrics = generate_metric_table(y_t_max, y_p_max, class_names)
    metric_fields = ("Metric", *MetricEntry._fields)

    roc_figure = plt_html(
        roc_plot(y_pred, y_test, n_classes), report_format=ReportFormatEnum.png, size=(800, 800)
    )

    model_name = "model_name"

    bundle = NOD(
        dict_of(title, model_name, confusion_matrix, metric_fields, metrics, predictions, roc_figure)
    )

    generate_html(file_name, **bundle)
    if do_generate_pdf:
        generate_pdf(file_name)


a()
