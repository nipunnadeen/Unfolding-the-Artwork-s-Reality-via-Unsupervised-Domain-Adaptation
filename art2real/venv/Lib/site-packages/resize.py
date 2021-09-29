#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resize all JEPG files in the given directory into one size.

Usage:
  resize.py <path> [--height=<px>] [--width=<px>]
  resize.py -h | --help
  resize.py --version

Options:
  -h --help           Show this screen.
  -v --version        Show version.
  -O --output-dir     Full path to directory with photos (./processed is default)
  -H --height=<px>    Set resulting height in pixels as number. [Default 1024]
  -W --width=<px>     Set resulting width in pixels as number. [Default 768]

"""
from __future__ import unicode_literals, print_function

import os
import logging

try:
    from PIL import Image
    from PIL import ImageOps
except ImportError:
    import Image
    import ImageOps
from docopt import docopt

logging.basicConfig(level=logging.INFO)
__version__ = "0.1.1"
__author__ = "Bartosz Dąbrowski"
__license__ = "MIT"


DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 768


def resize_image(image, to, height, width):
    """
    Preforms resize of given Image object which is expected to be with jpeg extension.

    Args:
        image: Image object.
        to: Full directory for output results.
        height: Height of the processed images.
        width: Width of the processed images.
    Returns:
        Absolute path to new image.
    """
    size = (height, width)
    oimage = ImageOps.fit(image, size, Image.ANTIALIAS)
    name, extension = os.path.basename(image.filename).split(os.extsep)
    new_name = ''.join([name, '_{}x{}'.format(height, width), os.extsep, extension])
    new_full_path = os.path.join(to, new_name)
    oimage.save(new_full_path, 'JPEG')
    return new_full_path


def main():
    """Main entry point for the resize CLI."""
    arguments = docopt(__doc__, version=__version__)
    print(arguments)
    output_path = arguments.get('--output-dir') or os.path.join(os.getcwd(), 'processed')
    try:
        os.makedirs(output_path)
    except OSError as error:
        if 'Errno 17' in str(error):
            # Dir already exists
            pass
        else:
            raise
    height = arguments.get('--height') or DEFAULT_HEIGHT
    width = arguments.get('--width') or DEFAULT_WIDTH

    for node in os.listdir(arguments['<path>']):
        filename = os.path.join(arguments['<path>'], node)
        logging.info('Processing file %s' % filename)
        if os.path.splitext(filename)[1].lower() not in ('.jpg', '.jpeg'):
            logging.info('{} is not jpeg - skipped'.format(filename))
            return
        image = Image.open(filename)
        new_full_path = resize_image(image, output_path, int(height), int(width))
        logging.info('Saved {}, Changed size: {} -> {} and {}MB to {}MB'
                     .format(new_full_path,
                             '{}x{}px'.format(image.size[0], image.size[1]),
                             '{}x{}px'.format(height, width),
                             os.stat(image.filename).st_size / 1000. / 1000.,
                             os.stat(new_full_path).st_size / 1000. / 1000.))


if __name__ == '__main__':
    main()
