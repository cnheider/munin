#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pathlib
from collections import namedtuple

import numpy as np

from draugr import plot_cf
from munin.utilities.html_embeddings import generate_math_html, plt_html, plt_html_svg

ReportEntry = namedtuple("ReportEntry", ("name", "figure", "prediction", "truth", "outcome"))

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""


def generate_html(file_name, template_page="classification_report_template.html", **kwargs):
    template_path = os.path.join(os.path.dirname(__file__), "templates")
    from jinja2 import Environment, select_autoescape, FileSystemLoader

    env = Environment(loader=FileSystemLoader(template_path), autoescape=select_autoescape(["html", "xml"]))
    template = env.get_template(template_page)
    with open(f"{file_name}.html", "w") as f:
        f.writelines(template.render(**kwargs))


def generate_pdf(file_name):
    import pdfkit

    pdfkit.from_file(f"{file_name}.html", f"{file_name}.pdf")


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (3, 3)
    from warg import NOD

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
    predictions = [[a, b, d], [a, c, d], [a, c, b], [c, b, e]]

    accuracy = generate_math_html("\dfrac{tp+tn}{N}"), [4, 4, 5], 4.2
    precision = generate_math_html("\dfrac{tp}{tp+fp}"), [4, 4, 5], 4.2
    recall = generate_math_html("\dfrac{tp}{tp+fn}"), [4, 4, 5], 5
    f1_score = generate_math_html("2*\dfrac{precision*recall}{precision+recall}"), [4, 4, 5], 4
    support = generate_math_html("N_{class_truth}"), [4, 4, 5], 6
    metrics = NOD.dict_of(accuracy, precision, f1_score, recall, support).as_flat_tuples()

    bundle = NOD.dict_of(title, confusion_matrix, metrics, predictions)

    file_name = title.lower().replace(" ", "_")

    generate_html(file_name, **bundle)
    # generate_pdf(file_name)
