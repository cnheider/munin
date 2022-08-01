#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

__doc__ = """
Created on 27/04/2019

@author: cnheider
"""


from typing import Sequence

import numpy
from pycm import ConfusionMatrix
from sorcery import dict_of
from warg.data_structures.named_ordered_dictionary import NOD

from munin.html_embeddings import MetricEntry, generate_math_html


def generate_metric_table(
    truths: Sequence, predictions: Sequence, categories: Sequence, decimals: int = 1
) -> Sequence[MetricEntry]:
    """

    :param truths:
    :param predictions:
    :param categories:
    :param decimals:
    :return:
    """

    cm = ConfusionMatrix(actual_vector=truths, predict_vector=predictions)
    cm.relabel({k: v for k, v in zip(range(len(categories)), categories)})

    support = MetricEntry(
        "Occurrence of each class (P)",
        generate_math_html("TP+FN"),
        {k: numpy.round(v, decimals) for k, v in cm.P.items()},
        numpy.round(sum(cm.P.values()) / len(categories), decimals),
    )

    sensitivity = MetricEntry(
        "True Positive Rate (TPR)",
        generate_math_html("\dfrac{TP}{TP+FN}"),
        {k: numpy.round(v, decimals) for k, v in cm.TPR.items()},
        numpy.round(sum(cm.TPR.values()) / len(categories), decimals),
    )

    specificity = MetricEntry(
        "True Negative Rate (TNR)",
        generate_math_html("\dfrac{TN}{TN+FP}"),
        {k: numpy.round(v, decimals) for k, v in cm.TNR.items()},
        numpy.round(sum(cm.TNR.values()) / len(categories), decimals),
    )

    precision = MetricEntry(
        "Positive Predictive Rate (PPV)",
        generate_math_html("\dfrac{TP}{TP+FP}"),
        {k: numpy.round(v, decimals) for k, v in cm.PPV.items()},
        numpy.round(sum(cm.PPV.values()) / len(categories), decimals),
    )

    npv = MetricEntry(
        "Negative Predictive Value (NPV)",
        generate_math_html("\dfrac{TP}{TP+FP}"),
        {k: numpy.round(v, decimals) for k, v in cm.NPV.items()},
        numpy.round(sum(cm.NPV.values()) / len(categories), decimals),
    )

    accuracy = MetricEntry(
        "Trueness",
        generate_math_html("\dfrac{TP+TN}{TP+TN+FP+FN}"),
        {k: numpy.round(v, decimals) for k, v in cm.ACC.items()},
        numpy.round(sum(cm.ACC.values()) / len(categories), decimals),
    )

    f1_score = MetricEntry(
        "Harmonic mean of precision and sensitivity",
        generate_math_html("2*\dfrac{PPV*TPR}{PPV+TPR}"),
        {k: numpy.round(v, decimals) for k, v in cm.F1.items()},
        numpy.round(sum(cm.F1.values()) / len(categories), decimals),
    )

    mcc = MetricEntry(
        "Matthews correlation coefficient",
        generate_math_html("\dfrac{TP*TN-FP*FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}"),
        {k: numpy.round(v, decimals) for k, v in cm.MCC.items()},
        numpy.round(sum(cm.MCC.values()) / len(categories), decimals),
    )

    roc_auc = MetricEntry(
        "Receiver Operating Characteristics (ROC), "
        "Sensitivity vs (1 − Specificity), "
        "(True Positive Rate vs False Positive Rate), "
        "Area Under the Curve (AUC)",
        generate_math_html("\dfrac{TNR+TPR}{2}"),
        {k: numpy.round(v, decimals) for k, v in cm.AUC.items()},
        numpy.round(sum(cm.AUC.values()) / len(categories), decimals),
    )

    return NOD(
        dict_of(
            support,
            sensitivity,
            specificity,
            precision,
            npv,
            accuracy,
            f1_score,
            mcc,
            roc_auc,
        )
    ).as_flat_tuples()
