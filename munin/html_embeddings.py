#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

from collections import namedtuple
from enum import Enum
from io import BytesIO, StringIO
from typing import Tuple

from matplotlib import pyplot
from matplotlib.figure import Figure

MetricEntry = namedtuple("MetricEntry", ("Description", "Math", "Values", "Aggregated"))


class ReportFormatEnum(Enum):
    jpg = "jpeg"
    html = "html"
    pdf = "pdf"
    svg = "svg"
    png = "png"
    jpeg = "jpeg"
    gif = "gif"
    tiff = "tiff"
    bmp = "bmp"
    svg_inline = "svg_inline"
    png_inline = "png_inline"
    jpeg_inline = "jpeg_inline"
    gif_inline = "gif_inline"
    tiff_inline = "tiff_inline"
    bmp_inline = "bmp_inline"


def generate_math_html(equation: str = "e^x", inline: bool = True, html_classes: str = "math_span") -> str:
    """
    For inline math, use \(...\).
    For standalone math, use $$...$$, \[...\] or \begin...\end.
    md = markdown.Markdown(extensions=['mdx_math'])
    md.convert('$$e^x$$')

    :param html_classes:
    :param equation:
    :param inline:
    :return:"""
    import markdown

    md = markdown.Markdown(extensions=["mdx_math"], extension_configs={"mdx_math": {"add_preview": True}})
    if inline:
        stripped = md.convert(f"\({equation}\)").lstrip("<p>").rstrip("</p>")
        return f'<span class="{html_classes}"><{stripped}></span>'
    return md.convert(f"$${equation}$$")


def plt_html_svg(fig: Figure = None, *, size: Tuple[int, int] = (400, 400), dpi: int = 100) -> str:
    """

    if figure not supplied it USEs lastest figure of pyplot

    :param fig:
    :param size:
    :param dpi:
    :return:
    """
    fig_file = StringIO()
    if fig is None:  # USE the latest figure
        pyplot.savefig(fig_file, format=ReportFormatEnum.svg.value, dpi=dpi)
    else:
        fig.savefig(fig_file, format=ReportFormatEnum.svg.value, dpi=dpi)
    return f'<svg width="{size[0]}" height="{size[1]}" {fig_file.getvalue().split("<svg")[1]}'


def plt_html(
    fig: Figure = None,
    *,
    title: str = "image",
    report_format: ReportFormatEnum = "png",
    size: Tuple[int, int] = (400, 400),
    dpi: int = 100,
) -> str:
    """

    if figure not supplied it USEs lastest figure of pyplot

    :rtype: object
    :param fig:
    :param title:
    :param report_format:
    :param size:
    :param dpi:
    :return:
    """
    report_format = ReportFormatEnum(report_format)

    if report_format == ReportFormatEnum.svg:
        return plt_html_svg(fig, size=size, dpi=dpi)

    import base64

    fig_file = BytesIO()
    if fig is None:  # USE lastest figure
        pyplot.savefig(fig_file, format=report_format.value, dpi=dpi)
    else:
        fig.savefig(fig_file, format=report_format.value, dpi=dpi)
    fig_file.seek(0)  # rewind to beginning of file
    b64_img = base64.b64encode(fig_file.getvalue()).decode("ascii")
    return (
        f"<img "
        f'width="{size[0]}" '
        f'height="{size[1]}" '
        f'src="data:image/{report_format.value};base64,{b64_img}" '
        f'alt="{title}"/><br>'
    )
