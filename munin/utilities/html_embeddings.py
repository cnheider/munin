#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import BytesIO, StringIO

import markdown

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

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
        return f"<{stripped}>"
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
