#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import matplotlib.pyplot as plt
from draugr import plot_cf
from warg import NOD

from munin.utilities.html_embeddings import plt_html_svg

plt.rcParams["figure.figsize"] = (3, 3)
import numpy as np

from munin.generate_report import generate_pdf, generate_html, ReportEntry
from munin.utilities.html_embeddings import generate_math_html, plt_html

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""


def test_generation(do_generate_pdf=False):
    data_path = pathlib.Path.home()

    num_classes = 3
    cell_width = (800 / num_classes) - 6 - 6 * 2

    plt.plot(np.random.random((3, 3)))

    a = ReportEntry(
        name=1, figure=plt_html_svg(size=[cell_width, cell_width]), prediction="a", truth="b", outcome="fp"
    )

    plt.plot(np.ones((9, 3)))

    b = ReportEntry(
        name=2,
        figure=plt_html(format="svg", size=[cell_width, cell_width]),
        prediction="b",
        truth="c",
        outcome="fp",
    )

    plt.plot(np.ones((5, 6)))

    c = ReportEntry(
        name=3, figure=plt_html(size=[cell_width, cell_width]), prediction="a", truth="a", outcome="tp"
    )

    d = ReportEntry(
        name="fas3",
        figure=plt_html(format="jpg", size=[cell_width, cell_width]),
        prediction="a",
        truth="a",
        outcome="tp",
    )

    e = ReportEntry(
        name="fas3",
        figure=plt_html(format="jpeg", size=[cell_width, cell_width]),
        prediction="c",
        truth="c",
        outcome="tn",
    )

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel="linear", C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    plot_cf(y_pred, y_test, class_names)

    title = "Classification Report"
    confusion_matrix = plt_html(format="png", size=[800, 800])
    entries = [[a, b, d], [a, c, d], [a, c, b], [c, b, e]]

    accuracy = generate_math_html("\dfrac{tp+tn}{N}"), [4, 4, 5], 4.2
    precision = generate_math_html("\dfrac{tp}{tp+fp}"), [4, 4, 5], 4.2
    recall = generate_math_html("\dfrac{tp}{tp+fn}"), [4, 4, 5], 5
    f1_score = generate_math_html("2*\dfrac{precision*recall}{precision+recall}"), [4, 4, 5], 4
    support = generate_math_html("N_{class_truth}"), [4, 4, 5], 6
    metrics = NOD.dict_of(accuracy, precision, f1_score, recall, support).as_flat_tuples()

    bundle = NOD.dict_of(title, confusion_matrix, metrics, entries)

    file_name = title.lower().replace(" ", "_")

    generate_html(file_name, **bundle)
    if do_generate_pdf:
        generate_pdf(file_name)


if __name__ == "__main__":
    test_generation(True)
