"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from __future__ import division
import codecs
import argparse
from collections import Counter
import gc
import torch
import random
import numpy as np
import pandas as pd
from rdkit import Chem
import onmt.inputters as inputters
import onmt.utils
import onmt.opts as opts
from onmt.utils.logging import logger
# from onmt.utils.optimizers import build_optim
# from onmt.models import build_model_saver
from onmt.inputters import EOS_WORD, PAD_WORD
from onmt.reinforce.scoring_functions import get_scoring_function
from onmt.utils.misc import tile
# from onmt.utils.loss import build_loss_compute

def build_translator_ms(opt, logger=None, out_file=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        # model (:obj:`onmt.models.NMTModel`): the model to train
        # fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')
        # out_file = codecs.open(opt.output, 'a', 'utf-8')

        # if opt.log_probs:
        #     log_probs_out_file = codecs.open(opt.output + '_log_probs', 'w+', 'utf-8')
    log_probs_out_file = None
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    train_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"].vocab, model_opt)

    # Build optimizer.
    # optim = build_optim(model, opt)
    optim = None

    # Build model saver
    # model_saver = build_model_saver(model_opt, opt, agent, fields, optim)

    norm_method = 'sents'
    grad_accum_count = 1
    # n_gpu = opt.world_size

    # trunc_size = opt.truncated_decoder  # Badly named...
    # shard_size = opt.max_generator_batches
    # gpu_verbose_level = opt.gpu_verbose_level

    n_gpu = 1
    if opt.gpu > -1:  # case 1 GPU only
        device_id = 0
    else:  # case only CPU
        device_id = -1
        # n_gpu = 0
    if device_id >= 0:
        # gpu_rank = opt.gpu_ranks[device_id]
        torch.cuda.set_device(device_id)
    # if opt.seed > 0:
    seed = 666
    torch.manual_seed(seed)
    random.seed(seed)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order

    # report_manager = onmt.utils.build_report_manager(opt)
    data_type = 'text'
    kwargs = {k: getattr(opt, k)
              for k in ["sample_rate", "gpu", "n_best", "window_size", "window_stride",
                        "window", "image_channel_size"]}
    for key in kwargs:
        print(str(key) + ':' + str(kwargs[key]))
    translator = Translator_ms(model, train_loss, optim, fields,
                           #     trunc_size,
                           # shard_size,
                               data_type, norm_method,
                               # n_gpu, gpu_rank,
                               # gpu_verbose_level,
                               report_manager=None,
                               # report_manager,
                               model_saver=None,out_file=out_file, **kwargs)


    return translator


class Translator_ms(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, optim, fields,
                 # trunc_size=0, shard_size=32,
                 data_type='text',
                 norm_method="sents", gpu=1, gpu_rank=1,
                 # gpu_verbose_level=0,
                 report_manager=None, model_saver=None,
                 out_file=None,
                 n_best=64,
                 sample_rate=16000,
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 image_channel_size=3
                 ):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.optim = optim
        self.fields = fields
        self.vocab = fields["tgt"].vocab
        # self.trunc_size = trunc_size
        # self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.gpu = gpu
        self.cuda = gpu > -1
        self.n_best = n_best
        # self.gpu_verbose_level = gpu_verbose_level
        # self.report_manager = report_manager
        # self.model_saver = model_saver
        self.out_file = out_file

        #load dataset
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.image_channel_size = image_channel_size




    def translate_ms(self, src_path, tgt_path, batch_size, opt, logger=None):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """

        info = logger.info if logger is not None else print
        debug = logger.debug if logger is not None else print
        info('Start training...')
        assert src_path is not None and tgt_path is not None
        # assert src_path is not None
        data = inputters. \
            build_dataset(self.fields,
                          self.data_type,
                          src_path=src_path,
                          src_data_iter=None,
                          tgt_path=tgt_path,
                          tgt_data_iter=None,
                          src_dir=None,
                          sample_rate=self.sample_rate,
                          window_size=self.window_size,
                          window_stride=self.window_stride,
                          window=self.window,
                          use_filter_pred=False,
                          image_channel_size=self.image_channel_size)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"


        info(f'cur device: {cur_device}')
        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=False, shuffle=False)


        for batch in data_iter:
            self.model.eval()
            self.model.generator.eval()

            ringnum = int(self.n_best/50)
            self.n_best = int(self.n_best/ringnum)
            for xx in range(ringnum):
                normalization = 0
                normalization += batch.batch_size

                src = inputters.make_features(batch, 'src', self.data_type)
                #tgt = inputters.make_features(batch, 'tgt')
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                elif self.data_type == 'audio':
                    src_lengths = batch.src_lengths
                else:
                    src_lengths = None

                # beam_size, batch_size = 2, 1

                enc_states, memory_bank, src_lengths \
                    = self.model.encoder(src, src_lengths)
                dec_states = self.model.decoder.init_decoder_state(
                    src, memory_bank, enc_states, with_cache=True)
                # Tile states and memory beam_size times.
                dec_states.map_batch_fn(
                    lambda state, dim: tile(state, self.n_best, dim=dim))
                if type(memory_bank) == tuple:
                    device = memory_bank[0].device
                    memory_bank = tuple(tile(m, self.n_best, dim=1) for m in memory_bank)
                else:
                    memory_bank = tile(memory_bank, self.n_best, dim=1)
                    device = memory_bank.device
                memory_lengths = tile(src_lengths, self.n_best)

                alive_seq = torch.full(
                    [opt.batch_size * self.n_best, 1],
                    self.vocab.stoi[inputters.BOS_WORD],
                    dtype=torch.long,
                    device=device)


                # Decoder forward.
                # max_length = 200
                agent_outputs = []
                finished = torch.zeros(self.n_best).byte()
                if torch.cuda.is_available():
                    finished = finished.cuda().view(-1, 1)
                # print(f'finished init: {finished.size(),finished}')
                for step in range(opt.max_length):
                    decoder_input = alive_seq[:, -1].view(1, -1, 1)
                    dec_out, dec_states, attn = self.model.decoder(
                        decoder_input,
                        memory_bank,
                        dec_states,
                        memory_lengths=memory_lengths,
                        step=step)

                    log_probs = self.model.generator.forward(dec_out.squeeze(0))

                    '''
                    just try
                    '''
                    # print(f'log_probs: {log_probs}')
                    # log_probs_sum = log_probs.sum()
                    agent_outputs += [log_probs]
                    probs = np.exp(log_probs.clone().detach().cpu().numpy())
                    # print(f'probs: {probs}')
                    x = torch.multinomial(self.toVariable(probs), 1).view(-1, 1)

                    x_new = x

                    alive_seq = torch.cat(
                        [alive_seq, x_new.view(-1, 1)], -1)
                    EOS_sampled = (x_new.data == self.vocab.stoi[inputters.EOS_WORD]).data
                    finished = torch.ge(finished + EOS_sampled, 1)
                    if torch.prod(finished) == 1: break

                # 4.change sequences to smiles
                self.seq_to_smiles(alive_seq)
                # 对目前sample的结果加权重，直接计数，/count
        self.out_file.close()




    # get multi smile
    def seq_to_smiles(self, seqs):
        """
            Takes an output sequence from the RNN and returns the
            corresponding SMILES.
            seqs:[batch, seq_lengths]
        """
        # print(type(seq))
        batch,len = seqs.size()

        smiles = []
        for i in range(batch):
            chars = []
            seq_lst = seqs[i, 1:].view(-1).cpu().numpy()
            for c in seq_lst:
                # print(i)
                if c == self.vocab.stoi[EOS_WORD]: break
                if c == self.vocab.stoi[PAD_WORD]: break
                chars.append(self.vocab.itos[c])
                # chars.append(' ')
            smile = " ".join(chars)
            print (smile)
            smiles.append(smile)
        self.out_file.write('\n'.join(smiles) + '\n')
        self.out_file.flush()
        # return smiles


    def toVariable(self, tensor):
        """Wrapper for torch.autograd.Variable that also accepts
           numpy arrays directly and automatically assigns it to
           the GPU. Be aware in case some operations are better
           left to the CPU."""
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor).cuda()
        return torch.autograd.Variable(tensor)



    def var(self, a):
        return torch.tensor(a, requires_grad=True)

    def rvar(self, a, repeat_size):
        return self.var(a.repeat(1, repeat_size, 1))