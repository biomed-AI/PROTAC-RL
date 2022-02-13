#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.reinforce.trainer_ms import build_rl_ms_trainer

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts
import torch


def main(opt, logger):
    RLmodel = build_rl_ms_trainer(opt, logger=logger)

    RLmodel.train(src_path=opt.src,
                  tgt_path=opt.tgt,
                  batch_size=opt.batch_size,
                  train_steps=opt.train_steps,
                  opt=opt,
                  logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train_agent_ms.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    onmt.opts.agent_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt, logger)