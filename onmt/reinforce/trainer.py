#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import math

import torch

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble
from onmt.reinforce import translation
from onmt.utils.optimizers import build_optim
from onmt.models import build_model_saver
import random


def build_rl_trainer(opt, report_score=True, logger=None, out_file=None, log_probs_out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

        if opt.log_probs:
            log_probs_out_file = codecs.open(opt.output + '_log_probs', 'w+', 'utf-8')

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)


    fields, prior, agent, model_opt, agent_checkpoint = \
        onmt.model_builder.load_rl_model(opt, dummy_opt.__dict__)


    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "data_type", "replace_unk", "gpu", "verbose", "fast",
                        "sample_rate", "window_size", "window_stride",
                        "window", "image_channel_size", "mask_from"]}

    trainer = RL_Trainer(prior, agent, agent_checkpoint, fields, opt, model_opt, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, logger=logger,
                            log_probs_out_file=log_probs_out_file, train_model=True,
                            **kwargs)
    return trainer


class RL_Trainer(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       agent agent (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 agent,
                 prior,
                 agent_checkpoint,
                 fields,
                 opt,
                 model_opt,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 logger=None,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate=16000,
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 log_probs_out_file=None,
                 fast=False,
                 mask_from='',
                 image_channel_size=3,
                 train_model=False):
        self.logger = logger
        self.gpu = gpu
        self.cuda = gpu > -1

        self.agent = agent
        self.prior = prior
        self.agent_checkpoint = agent_checkpoint
        self.fields = fields
        self.opt = opt
        self.model_opt = model_opt
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.log_probs_out_file = log_probs_out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.fast = fast
        self.image_channel_size = image_channel_size
        self.train_model = train_model

        if mask_from != '':
            from ..utils.masking import ChemVocabMask
            self.mask = ChemVocabMask(from_file=mask_from)
        else:
            self.mask = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def train(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters. \
            build_dataset(self.fields,
                          self.data_type,
                          src_path=src_path,
                          src_data_iter=src_data_iter,
                          tgt_path=tgt_path,
                          tgt_data_iter=tgt_data_iter,
                          src_dir=src_dir,
                          sample_rate=self.sample_rate,
                          window_size=self.window_size,
                          window_stride=self.window_stride,
                          window=self.window,
                          use_filter_pred=self.use_filter_pred,
                          image_channel_size=self.image_channel_size)


        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=False, shuffle=False)

        builder = translation.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        all_scores = []
        all_predictions = []

        if len(self.opt.models) > 1:
            optim = build_optim(self.agent, self.opt, self.agent_checkpoint)
            step = optim._step + 1
        else:
            optim = build_optim(self.agent, self.opt)
            step = 0

        model_saver = build_model_saver(self.model_opt, self.opt, self.agent, self.fields, optim)
        print (f'init step: {step}')
        for batch in data_iter:

            # for step in range(self.opt.train_steps):
            while step <= self.opt.train_steps:
                batch_data = self.translate_batch(batch, data, self.train_model, fast=self.fast)
                translations = builder.from_batch(batch_data, self.opt)

                prior_likelihood = []
                agent_likelihood = []
                scores = []

                for trans in translations:
                    all_scores += [trans.agent_scores[:self.n_best]]

                    n_best_preds = [" ".join(pred)
                                    for pred in trans.pred_sents[:self.n_best]]
                    all_predictions += [n_best_preds]
                    self.out_file.write('\n'.join(n_best_preds) + '\n')
                    self.out_file.flush()

                    if self.log_probs_out_file is not None:
                        self.log_probs_out_file.write(
                            '\n'.join([str(t.item()) for t in trans.agent_scores[:self.n_best]]) + '\n')
                        self.log_probs_out_file.flush()

                    agent_likelihood = torch.stack(trans.agent_scores)[:self.n_best]
                    prior_likelihood = trans.prior_scores.to(agent_likelihood.device)
                    scores = torch.tensor(trans.scores).to(agent_likelihood.device)
                    augmented_likelihood = prior_likelihood + self.opt.sigma * scores

                    loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                    loss = loss.mean()

                    self.agent.zero_grad()
                    loss.backward()
                    optim.step()

                    if self.verbose and step % self.opt.report_every == 0:
                        # sent_number = next(counter)
                        output = trans.log()
                        if self.logger:
                            self.logger.info("STEP ---- {}".format(step))
                            self.logger.info("Learning Rate: ".format(self.opt.learning_rate))
                            self.logger.info("MEAN LOSS: {}".format(loss.item()))
                            self.logger.info("MEAN SCORE: {}".format(scores.mean().item()))
                            self.logger.info(output)
                        else:
                            os.write(1, output.encode('utf-8'))

                    model_saver.maybe_save(step+1)

                    # TODO: now is only for one batch with one molecule
                    # prior_likelihood.append(trans.prior_scores)
                    # agent_likelihood.append(torch.stack(trans.agent_scores))
                    # scores.append(torch.tensor(trans.scores))

                    # Debug attention.
                    if attn_debug:
                        preds = trans.pred_sents[0]
                        preds.append('</s>')
                        attns = trans.attns[0].tolist()
                        if self.data_type == 'text':
                            srcs = trans.src_raw
                        else:
                            srcs = [str(item) for item in range(len(attns[0]))]
                        header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                        output = header_format.format("", *srcs) + '\n'
                        for word, row in zip(preds, attns):
                            max_index = row.index(max(row))
                            row_format = row_format.replace(
                                "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                            row_format = row_format.replace(
                                "{:*>10.7f} ", "{:>10.7f} ", max_index)
                            output += row_format.format(word, *row) + '\n'
                            row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                        os.write(1, output.encode('utf-8'))

                step = step + 1
                # if self.report_score:
                #     msg = self._report_score('PRED', agent_score_total,
                #                              agent_words_total)
                #     if self.logger:
                #         self.logger.info(msg)
                #     else:
                #         print(msg)
                #     if tgt_path is not None:
                #         msg = self._report_score('GOLD', gold_score_total,
                #                                  gold_words_total)
                #         if self.logger:
                #             self.logger.info(msg)
                #         else:
                #             print(msg)
                #         if self.report_bleu:
                #             msg = self._report_bleu(tgt_path)
                #             if self.logger:
                #                 self.logger.info(msg)
                #             else:
                #                 print(msg)
                #         if self.report_rouge:
                #             msg = self._report_rouge(tgt_path)
                #             if self.logger:
                #                 self.logger.info(msg)
                #             else:
                #                 print(msg)



        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores, all_predictions

    def translate_batch(self, batch, data, train_model, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        if train_model:
            if fast:
                return self._fast_translate_batch(
                    batch,
                    data,
                    self.max_length,
                    min_length=self.min_length,
                    n_best=self.n_best,
                    return_attention=self.replace_unk)
            else:
                return self._translate_batch(batch, data)

        else:
            with torch.no_grad():
                if fast:
                    return self._fast_translate_batch(
                        batch,
                        data,
                        self.max_length,
                        min_length=self.min_length,
                        n_best=self.n_best,
                        return_attention=self.replace_unk)
                else:
                    return self._translate_batch(batch, data)

    def _fast_translate_batch(self,
                              batch,
                              data,
                              max_length,
                              min_length=0,
                              n_best=1,
                              return_attention=False):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert data.data_type == 'text'
        assert not self.copy_attn
        assert not self.dump_beam
        assert not self.use_filter_pred
        assert self.block_ngram_repeat == 0
        assert self.global_scorer.beta == 0

        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab
        start_token = vocab.stoi[inputters.BOS_WORD]
        end_token = vocab.stoi[inputters.EOS_WORD]

        # Encoder forward.
        src = inputters.make_features(batch, 'src', data.data_type)
        _, src_lengths = batch.src
        enc_states, memory_bank, src_lengths \
            = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if type(memory_bank) == tuple:
            device = memory_bank[0].device
            memory_bank = tuple(tile(m, beam_size, dim=1) for m in memory_bank)
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
            device = memory_bank.device
        memory_lengths = tile(src_lengths, beam_size)

        top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        batch_offset = torch.arange(batch_size, dtype=torch.long)

        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=device)
        alive_attn = None

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        if self.mask is not None:
            mask = self.mask.get_log_probs_masking_tensor(src.squeeze(2), beam_size).to(memory_bank.device)


        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1, 1)

            # Decoder forward.
            dec_out, dec_states, attn = self.model.decoder(
                decoder_input,
                memory_bank,
                dec_states,
                memory_lengths=memory_lengths,
                step=step)

            # Generator forward.
            log_probs = self.model.generator.forward(dec_out.squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, end_token] = -1e20

            if self.mask is not None:
                log_probs = log_probs * mask

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            if return_attention:
                current_attn = attn["std"].index_select(1, select_indices)
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = alive_attn.index_select(1, select_indices)
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)

            # Save finished hypotheses.
            if is_finished.any():
                # Penalize beams that finished.
                topk_log_probs.masked_fill_(is_finished, -1e10)
                is_finished = is_finished.to('cpu')
                top_beam_finished |= is_finished[:, 0].eq(1)

                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                attention = (
                    alive_attn.view(
                        alive_attn.size(0), -1, beam_size, alive_attn.size(-1))
                    if alive_attn is not None else None)
                non_finished_batch = []
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        # if (predictions[i, j, 1:] == end_token).sum() <= 1:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:],  # Ignore start_token.
                            attention[:, i, j, :memory_lengths[i]]
                            if attention is not None else None))
                    # End condition is the top beam finished and we can return
                    # n_best hypotheses.
                    if top_beam_finished[i] and len(hypotheses[b]) >= n_best:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred, attn) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["attention"][b].append(
                                attn if attn is not None else [])
                    else:
                        non_finished_batch.append(i)
                non_finished = torch.tensor(non_finished_batch)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                top_beam_finished = top_beam_finished.index_select(
                    0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                non_finished = non_finished.to(topk_ids.device)
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                select_indices = batch_index.view(-1)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                if alive_attn is not None:
                    alive_attn = attention.index_select(1, non_finished) \
                        .view(alive_attn.size(0),
                              -1, alive_attn.size(-1))

            # Reorder states.
            if type(memory_bank) == tuple:
                memory_bank = tuple(m.index_select(1, select_indices) for m in memory_bank)
            else:
                memory_bank = memory_bank.index_select(1, select_indices)
            memory_lengths = memory_lengths.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

            if self.mask is not None:
                mask = mask.index_select(0, select_indices)

        return results

    def _translate_batch(self, batch, data):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a, grad=False):
            return a.clone().detach().requires_grad_(grad)

        def rvar(a, grad=False):
            if grad:
                return a.repeat(1, beam_size, 1)
            else:
                return var(a.detach().repeat(1, beam_size, 1), grad)

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m, grad):
            if grad:
                return m.view(beam_size, batch_size, -1)
            else:
                return m.detach().view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = inputters.make_features(batch, 'src', data_type)

        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        enc_states, memory_bank, src_lengths \
            = self.agent.encoder(src, src_lengths)
        dec_states = self.agent.decoder.init_decoder_state(
            src, memory_bank, enc_states)

        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data) \
                .long() \
                .fill_(memory_bank.size(0))

        if self.mask is not None:
            mask = self.mask.get_log_probs_masking_tensor(src.squeeze(2), 1).to(memory_bank.device)

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(rvar(x, True) for x in memory_bank)
        else:
            memory_bank = rvar(memory_bank, True)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.agent.decoder(
                inp, memory_bank, dec_states,
                memory_lengths=memory_lengths,
                step=i)

            dec_out = dec_out.squeeze(0)

            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.agent.generator.forward(dec_out)
                out = unbottle(out, True)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"], True)
            else:
                out = self.agent.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out, True),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"], True)

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                if not b.done():
                    if self.mask is not None:
                        b.advance(out[:, j],
                                  beam_attn.data[:, j, :memory_lengths[j]], mask[j])
                    else:
                        b.advance(out[:, j],
                                  beam_attn.data[:, j, :memory_lengths[j]])
                    dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        # with torch.no_grad():
        ret["prior_score"] = self._run_target_prior(batch, data, self.prior, ret["predictions"], beam_size)
        ret["batch"] = batch

        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    """
    batch:  tuple(src, tgt, src_map, aligment..five)
    pred:   list [1,beam_size,tgt_len-1]
    batch_size: n_best
    """

    def _run_target_prior(self, batch, data, prior, pred, beam_size):
        # print (f'batch:{batch}')
        # print (f'pred:{len(pred), len(pred[0]), len(pred[0][0])}')
        # print (f'pred[0][0]:{pred[0][0]}')
        # predlenlst = []
        # for xx in range(len(pred[0])):
        #     predlenlst.append(len(pred[0][xx]))
        # print (f'pred seqs lens:{predlenlst}')

        # testseq = []
        # for xx in range(len(pred[0][0])):
        #     testseq.append(pred[0][0][xx])
        # vocab = self.fields["tgt"].vocab
        # test = ''.join([vocab.itos[i] for i in testseq])
        # print(f'pred: {test}')


        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        else:
            src_lengths = None
        src = inputters.make_features(batch, 'src', data_type)
        # seq , batch , word
        input_src_lengths = src_lengths.repeat_interleave(self.n_best, dim=0)

        input_src = torch.repeat_interleave(src, self.n_best, dim=1)
        # print (f'n_best beam_size: {self.n_best, self.beam_size}')
        # print (f'batch.src: {input_src.size()}')
        tgt_pad = self.fields["tgt"].vocab.stoi[inputters.PAD_WORD]
        tgt_eos = self.fields["tgt"].vocab.stoi[inputters.EOS_WORD]
        tgt_bos = self.fields["tgt"].vocab.stoi[inputters.BOS_WORD]

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # batch , seq , beam

        tgt_complete = torch.stack([torch.nn.utils.rnn.pad_sequence( \
            [torch.stack([torch.tensor(tgt_bos, device=device)] + bm) \
             for bm in bt], padding_value=tgt_pad) for bt in pred])

        x, y, z = tgt_complete.size()

        # testseq = []
        # for xx in range(len(tgt_complete[0])):
        #     testseq.append(tgt_complete[0][xx][0])
        # # print (f'testseq: {testseq}')
        # print (x,y,z)
        # # test = list(tgt_complete[0, :, 0].view(-1).cpu().numpy())
        # test = ''.join([vocab.itos[i] for i in testseq])
        # print(f'tgt_complete: {test}')
        # print (f'chars: BOS: {tgt_bos}; EOS: {tgt_eos}; PAD: {tgt_pad}')
        # test = tgt_complete.reshape(x,z,y)
        # print (f'test.size: {test.size()}')
        # print (f'test:{test[0][0],len(test[0][0])}')
        # print (f'test:{test}')
        tgt_complete = tgt_complete.reshape(y, x * z, 1)

        tgt_in = tgt_complete[:-1, :, :]
        tgt_compare = tgt_complete[1:, :, :]

        # testseq = []
        #         # for xx in range(len(tgt_in)):
        #         #     testseq.append(tgt_in[xx][0][0])
        #         # # print (f'testseq: {testseq}')
        #         # # test = list(tgt_in[:, 0].view(-1).cpu().numpy())
        #         # test = ''.join([vocab.itos[i] for i in testseq])
        #         # print (f'tgt_in: {test}')
        #         # print (f'tgt_in.size: {tgt_in.size()}')


        # (1) run the encoder on the src
        enc_states, memory_bank, src_lengths \
            = self.prior.encoder(input_src, input_src_lengths)
        dec_states = \
            self.prior.decoder.init_decoder_state(input_src, memory_bank, enc_states)
        #  (2)compute the 'priorScore'
        #  (i.e. log likelihood) of the target under the prior
        tt = torch.cuda if self.cuda else torch
        prior_scores = tt.FloatTensor(batch.batch_size * self.n_best).fill_(0)
        dec_out, _, _ = self.prior.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=input_src_lengths)
        # print (f'dec_out.size: {dec_out.size()}')
        for dec, tgt in zip(dec_out, tgt_compare.data):
            # Log prob of each word.
            out = self.prior.generator.forward(dec)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            prior_scores += scores.view(-1)
        prior_scores = prior_scores.reshape(batch.batch_size, self.n_best)
        return prior_scores

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total,
                name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (base_dir, tgt_path),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        msg = res.strip()
        return msg
