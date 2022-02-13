#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator_ms import build_translator_ms

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import torch


def main(opt):
    translator = build_translator_ms(opt)
    translator.translate_ms(src_path=opt.src,
                             tgt_path=opt.tgt,
                             batch_size=opt.batch_size,
                             opt=opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate_ms.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    # onmt.opts.agent_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
