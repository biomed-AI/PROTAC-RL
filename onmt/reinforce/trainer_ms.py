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
from onmt.utils.optimizers import build_optim
from onmt.models import build_model_saver
from onmt.inputters import EOS_WORD, PAD_WORD
from onmt.reinforce.scoring_functions import get_scoring_function
from onmt.utils.misc import tile
# from onmt.utils.loss import build_loss_compute

def build_rl_ms_trainer(opt, logger=None, out_file=None, log_probs_out_file=None):
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

        if opt.log_probs:
            log_probs_out_file = codecs.open(opt.output + '_log_probs', 'w+', 'utf-8')

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, prior, agent, model_opt, agent_checkpoint = \
        onmt.model_builder.load_rl_model(opt, dummy_opt.__dict__)

    train_loss = onmt.utils.loss.build_loss_compute(
        agent, fields["tgt"].vocab, model_opt)

    # Build optimizer.
    if len(opt.models) > 1:
        optim = build_optim(agent, opt, agent_checkpoint)
    else:
        optim = build_optim(agent, opt)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, agent, fields, optim)

    norm_method = 'sents'
    grad_accum_count = 1
    # n_gpu = opt.world_size

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    gpu_verbose_level = opt.gpu_verbose_level

    n_gpu = len(opt.gpu_ranks)
    if n_gpu == 1:  # case 1 GPU only
        device_id = 0
    else:  # case only CPU
        device_id = -1
        # n_gpu = 0
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
        torch.cuda.set_device(device_id)
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)

    # report_manager = onmt.utils.build_report_manager(opt)
    data_type = 'text'
    kwargs = {k: getattr(opt, k)
              for k in ["sample_rate", "window_size", "window_stride",
                        "window", "image_channel_size", "scoring_function",
                        "score_function_num_processes"]}
    for key in kwargs:
        print(str(key) + ':' + str(kwargs[key]))
    trainer = RL_ms_Trainer(prior, agent, train_loss, optim, fields, trunc_size,
                           shard_size, data_type, norm_method,
                           n_gpu, gpu_rank,
                           gpu_verbose_level,
                           report_manager=None,
                           # report_manager,
                           model_saver=model_saver, **kwargs)


    return trainer


class RL_ms_Trainer(object):
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

    def __init__(self, prior, agent, train_loss, optim, fields,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                 sample_rate=16000,
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 image_channel_size=3,
                 scoring_function='SIM_3D',
                 score_function_num_processes=0,
                 ):
        # Basic attributes.
        # self.model = model
        self.prior = prior
        self.agent = agent
        self.train_loss = train_loss
        self.optim = optim
        self.fields = fields
        self.vocab = fields["tgt"].vocab
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver

        #load dataset
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.image_channel_size = image_channel_size

        self.scoring_function = scoring_function
        self.score_function_num_processes = score_function_num_processes

        # assert grad_accum_count > 0
        # if grad_accum_count > 1:
        #     assert(self.trunc_size == 0), \
        #         """To enable accumulated gradients,
        #            you must disable target sequence truncating."""

        # Set model in training mode.
        # self.model.train()

    def train(self, src_path, tgt_path, train_steps, batch_size, opt, logger=None):
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

        # if self.cuda:
        #     cur_device = "cuda"
        # else:
        #     cur_device = "cpu"

        if self.n_gpu > 0:
            cur_device = "cuda"
        else:
            cur_device = "cpu"
        info(f'cur device: {cur_device}')
        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=False, shuffle=False)

        cur_step = self.optim._step


        for batch in data_iter:

            scoring_function_kwargs = {}

            if opt.src_type == 'N':
                scoring_function_kwargs['src'] = self.seq_to_smile_tol(batch.src[0])
            else:
                scoring_function_kwargs['src'] = self.seq_to_smile(batch.src[0])
            scoring_function_kwargs['goal'] = opt.goal
            scoring_function_kwargs['ref'] = self.seq_to_smile(batch.tgt)
            scoring_function_kwargs['k'] = opt.score_para_k
            scoring_function_kwargs['w'] = opt.score_para_w
            scoring_function_kwargs['clf_path'] = opt.score_para_clf
            scoring_function = get_scoring_function(scoring_function=self.scoring_function,
                                                    num_processes=self.score_function_num_processes,
                                                    **scoring_function_kwargs
                                                    )
            info(scoring_function_kwargs['src'])
            info(scoring_function_kwargs['ref'])


            while cur_step <= train_steps:
                self.prior.eval()
                self.agent.eval()
                self.prior.generator.eval()
                self.agent.generator.eval()

                # smile_lst = []
                normalization = 0
                if self.norm_method == "tokens":
                    num_tokens = batch.tgt[1:].ne(
                        self.train_loss.padding_idx).sum()
                    normalization += num_tokens.item()
                else:
                    normalization += batch.batch_size

                src = inputters.make_features(batch, 'src', self.data_type)
                tgt = inputters.make_features(batch, 'tgt')
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                elif self.data_type == 'audio':
                    src_lengths = batch.src_lengths
                else:
                    src_lengths = None

                # beam_size, batch_size = 2, 1

                enc_states, memory_bank, src_lengths \
                    = self.agent.encoder(src, src_lengths)
                dec_states = self.agent.decoder.init_decoder_state(
                    src, memory_bank, enc_states, with_cache=True)
                # Tile states and memory beam_size times.
                dec_states.map_batch_fn(
                    lambda state, dim: tile(state, opt.n_best, dim=dim))
                if type(memory_bank) == tuple:
                    device = memory_bank[0].device
                    memory_bank = tuple(tile(m, opt.n_best, dim=1) for m in memory_bank)
                else:
                    memory_bank = tile(memory_bank, opt.n_best, dim=1)
                    device = memory_bank.device
                memory_lengths = tile(src_lengths, opt.n_best)

                alive_seq = torch.full(
                    [opt.batch_size * opt.n_best, 1],
                    self.vocab.stoi[inputters.BOS_WORD],
                    dtype=torch.long,
                    device=device)


                # Decoder forward.
                # max_length = 200
                agent_outputs = []
                finished = torch.zeros(opt.n_best).byte()
                if torch.cuda.is_available():
                    finished = finished.cuda().view(-1, 1)

                for step in range(opt.max_length):
                    decoder_input = alive_seq[:, -1].view(1, -1, 1)

                    dec_out, dec_states, attn = self.agent.decoder(
                        decoder_input,
                        memory_bank,
                        dec_states,
                        memory_lengths=memory_lengths,
                        step=step)

                    # Generator forward.
                    log_probs = self.agent.generator.forward(dec_out.squeeze(0))

                    '''
                    just try
                    '''

                    agent_outputs += [log_probs]
                    probs = np.exp(log_probs.clone().detach().cpu().numpy())
                    x = torch.multinomial(self.toVariable(probs), 1).view(-1, 1)

                    x_new = x

                    if opt.pred_rewrite:
                        x_new = self.overwrite_prediction(alive_seq[:, -1], x)
                    else:
                        x_new = x

                    alive_seq = torch.cat(
                        [alive_seq, x_new.view(-1, 1)], -1)
                    EOS_sampled = (x_new.data == self.vocab.stoi[inputters.EOS_WORD]).data
                    finished = torch.ge(finished + EOS_sampled, 1)

                    if torch.prod(finished) == 1: break

                # 2.calculate likelihood between outputs_p and sampled_groundtruth
                '''
                    agent_scores : (seq_length-1,batch_size, num_classes) *Log probabilities of each class*
                    agent_target: (seq_length-1,batch_size) *Target class index*
                    # permute unsqueeze()
                '''
                if type(agent_outputs) == list:
                    agent_outputs = torch.stack(agent_outputs)

                sequences = alive_seq.permute(1, 0)
                agent_target = sequences[1:]
                agent_scores = agent_outputs
                agent_gtruth = agent_target
                agent_likelihood = self.criterion_per_sample(agent_scores, agent_gtruth)

                # 3.change sequences to smiles
                smiles_sequences = self.seq_to_smiles(alive_seq)
                score = scoring_function(smiles_sequences)

                sequences = alive_seq.permute(1, 0)
                # 4.use prior to get the samples likelihood ,just like teacher force
                prior_tgt = sequences.unsqueeze(-1)[:-1]  # exclude last target from inputs
                prior_enc_final, prior_memory_bank, prior_lengths = self.prior.encoder(src, src_lengths)
                prior_enc_state = \
                    self.prior.decoder.init_decoder_state(src, prior_memory_bank, prior_enc_final)
                prior_enc_state.map_batch_fn(
                    lambda state, dim: tile(state, opt.n_best, dim=dim))
                if type(prior_memory_bank) == tuple:
                    prior_memory_bank = tuple(tile(m, opt.n_best, dim=1) for m in prior_memory_bank)
                else:
                    prior_memory_bank = tile(prior_memory_bank, opt.n_best, dim=1)
                prior_lengths = tile(prior_lengths, opt.n_best)
                prior_outputs, prior_states, prior_attns = \
                    self.prior.decoder(prior_tgt, prior_memory_bank,
                                 prior_enc_state,
                                 memory_lengths=prior_lengths)
                prior_outputs = self.prior.generator(prior_outputs)
                prior_scores = prior_outputs
                prior_likelihood = self.criterion_per_sample(prior_scores, agent_gtruth)
                new_score = score


                # 5.linear the two loss
                rb_score = new_score - new_score.mean()
                augmented_likelihood = prior_likelihood + opt.sigma * self.toVariable(new_score)
                loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

                self.agent.zero_grad()
                loss = loss.mean()
                loss.backward()
                self.optim.step()
                # smile_lst.extend(smiles_sequences)

                if cur_step % opt.report_every == 0:
                    info(f"Step ------ {cur_step};")
                    info(
                        f'loss :::  {loss.data},  score: {score.mean()}, lr: {self.optim.learning_rate, self.optim.original_lr} ')
                    info(
                        f'agent_likelihood:{agent_likelihood.data.mean()}, augmented_likelihood: {augmented_likelihood.data.mean()},   prior_likelihood:{prior_likelihood.data.mean()} ')

                    info(
                        f"Step {cur_step};Fraction valid SMILES: {self.fraction_valid_smiles(smiles_sequences) * 100:4.1f}")
                    info('samples:')
                    for i in range(opt.n_best):
                        info(smiles_sequences_standard[i])

                    del smiles_sequences
                    del smiles_sequences_standard
                    gc.collect()

                if cur_step % 50 == 0:
                    experience.print_memory()

                del enc_states,memory_bank,dec_states,alive_seq,\
                    dec_out, attn,agent_target,agent_scores,\
                    prior_enc_final, prior_memory_bank,prior_outputs, prior_states, prior_attns,\
                    weights
                gc.collect()

                self.model_saver.maybe_save(cur_step)
                cur_step += 1
                if cur_step > train_steps:
                    break

    def to_standardSmiles(self, sequences):
        standardSeqs = []
        for seq in sequences:
            mol = Chem.MolFromSmiles(seq)
            if mol is None:
                standardSeqs.append(seq)
            else:
                standardSeqs.append(Chem.MolToSmiles(mol))
        return standardSeqs
    

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    # get single smile
    def seq_to_smile(self, seq):
        """
            Takes an output sequence from the RNN and returns the
            corresponding SMILES.
            seqs:[seq_lengths,batch]

        """
        chars = []
        # print(type(seq))
        seq_lst = seq[1:, :].view(-1).cpu().numpy()
        for i in seq_lst:
            # print(i)
            if i == self.vocab.stoi[EOS_WORD]: break
            if i == self.vocab.stoi[PAD_WORD]: break
            chars.append(self.vocab.itos[i])
        smile = "".join(chars)
        return smile

    # get single smile
    def seq_to_smile_tol(self, seq):
        """
            Takes an output sequence from the RNN and returns the
            corresponding SMILES.
            seqs:[seq_lengths,batch]

        """
        chars = []
        # print(type(seq))
        seq_lst = seq.view(-1).cpu().numpy()
        for i in seq_lst:
            # print(i)
            if i == self.vocab.stoi[EOS_WORD]: break
            if i == self.vocab.stoi[PAD_WORD]: break
            chars.append(self.vocab.itos[i])
        smile = "".join(chars)
        return smile

    # get single smile
    def seq_to_smile_tol_test(self, seq):
        """
            Takes an output sequence from the RNN and returns the
            corresponding SMILES.
            seqs:[seq_lengths,batch]

        """
        chars = []
        # print(type(seq))
        seq_lst = seq.view(-1).cpu().numpy()
        for i in seq_lst:
            # print(i)
            chars.append(self.vocab.itos[i])
        smile = "".join(chars)
        return smile

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
            smile = "".join(chars)
            smiles.append(smile)
        return smiles

    # get multi smile
    def seq_to_smiles_tttol(self, seqs):
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
            seq_lst = seqs[i, :].view(-1).cpu().numpy()
            for c in seq_lst:
                # print(i)
                # if c == self.vocab.stoi[EOS_WORD]: break
                # if c == self.vocab.stoi[PAD_WORD]: break
                chars.append(self.vocab.itos[c])
            smile = "".join(chars)
            smiles.append(smile)
        return smiles


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

    def overwrite_prediction(self, prev_seq, pred_out):
        '''
        :param prev_seq: [batchsize]
        :param pred_out: [batchsize,1]
        :param vocab:
        :return:
        '''
        # print(f'overwrite: {prev_seq.size(),pred_out.size()}')
        curinput = pred_out.clone().detach()
        # print('ceshi :',self.vocab.stoi[inputters.EOS_WORD],self.vocab.stoi[inputters.PAD_WORD]) 3 1
        for index in range(prev_seq.size(0)):
            if prev_seq[index] == self.vocab.stoi[inputters.EOS_WORD] or prev_seq[index] == self.vocab.stoi[inputters.PAD_WORD]:
                # print('chenggong')
                curinput[index, 0] = self.vocab.stoi[inputters.PAD_WORD]
        # print(pred_step,curinput)
        # curinput = curinput.unsqueeze(0)
        # print(curinput.size())
        return curinput

    def criterion_per_sample(self, inputs, targets):
        """
            Custom Negative Log Likelihood loss that returns loss per example,
            rather than for the entire batch.
            NLLLoss
            Args:
                inputs : (seq_length,batch_size, num_classes) *Log probabilities of each class*
                targets: (seq_length,batch_size) *Target class index*

            Outputs:
                loss : (batch_size) *Loss for each example*
        """
        assert inputs.dim() == 3 and targets.dim() == 2
        assert inputs.size(0) == targets.size(0) and inputs.size(1) == targets.size(1)
        if torch.cuda.is_available():
            target_expanded = torch.zeros(inputs.size()).cuda()
        else:
            target_expanded = torch.zeros(inputs.size())
        
        targets = targets.unsqueeze(-1).contiguous().data

        target_expanded.scatter_(-1, targets, 1.0)
        non_pad_inputs =  (targets != self.vocab.stoi[PAD_WORD])

        loss = self.toVariable(target_expanded) * inputs * non_pad_inputs
        loss = torch.sum(loss, dim=-1) # (seq_length,batch_size)
        loss = torch.sum(loss,dim=0)  #([batch_size])
        return loss

    # get fraction valid smiles percentage
    def fraction_valid_smiles(self, smiles):
        """Takes a list of SMILES and returns fraction valid."""
        i = 0
        for smile in smiles:

            if Chem.MolFromSmiles(smile):
                i += 1
        return i / len(smiles)

    def var(self, a):
        return torch.tensor(a, requires_grad=True)

    def rvar(self, a, repeat_size):
        return self.var(a.repeat(1, repeat_size, 1))
