#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.reinforce.trainer import build_rl_trainer

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import torch


def main(opt, logger):
    RLmodel = build_rl_trainer(opt, logger=logger, report_score=True)

    RLmodel.train(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train_agent.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    onmt.opts.agent_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt, logger)