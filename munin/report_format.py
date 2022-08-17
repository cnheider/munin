#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/4/22
           """

__all__ = ["ReportFormatEnum"]

from enum import Enum


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
