""" Translation main class """
from __future__ import division, unicode_literals
from __future__ import print_function

import torch
import onmt.inputters as inputters
from onmt.reinforce.scoring_functions import get_scoring_function


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False):
        self.data = data
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == inputters.EOS_WORD:
                tokens = tokens[:-1]
                break
        if self.replace_unk and (attn is not None) and (src is not None):
            for i in range(len(tokens)):
                if tokens[i] == vocab.itos[inputters.UNK]:
                    _, max_index = attn[i].max(0)
                    tokens[i] = src_raw[max_index.item()]
        return tokens

    def from_batch(self, translation_batch, opt):
        batch = translation_batch["batch"]
        assert(len(translation_batch["prior_score"]) ==
               len(translation_batch["predictions"]))

        batch_size = batch.batch_size

        preds, agent_score, attn, prior_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["prior_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices.data)
        data_type = self.data.data_type
        if data_type == 'text':
            src = batch.src[0].data.index_select(1, perm)
        else:
            src = None

        if self.has_tgt:
            tgt = batch.tgt.data.index_select(1, perm)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            if data_type == 'text':
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src
            else:
                src_vocab = None
                src_raw = None
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)

            translation = Translation(src[:, b] if src is not None else None,
                                      src_raw, pred_sents,
                                      attn[b], agent_score[b], gold_sent,
                                      prior_score[b], opt)
            translations.append(translation)

        return translations


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, src, src_raw, pred_sents,
                 attn, agent_scores, tgt_sent, prior_scores, opt):
        self.src = src
        self.src_raw_without_L = ""
        self.src_raw = ""
        for i, token in enumerate(src_raw):
            if i == 0:
                self.src_raw += token
                continue
            self.src_raw += token
            self.src_raw_without_L += token
        self.pred_sents = []
        for sent in pred_sents:
            mol = ""
            for token in sent:
                mol += token
            self.pred_sents.append(mol)
        self.attns = attn
        self.agent_scores = agent_scores
        self.gold_sent = ""
        for token in tgt_sent:
            self.gold_sent += token
        self.prior_scores = prior_scores

        scoring_function_kwargs = {}
        if opt.src_type == 'N':
            self.src_raw_without_L = self.src_raw
        scoring_function_kwargs['src'] = self.src_raw_without_L
        scoring_function_kwargs['ref'] = self.gold_sent
        # scoring_function = get_scoring_function(scoring_function=opt.scoring_function,
        #                                         num_processes=opt.score_function_num_processes,
        #                                         **scoring_function_kwargs
        #                                         )
        self.scoring_function = get_scoring_function(scoring_function=opt.scoring_function,
                                                num_processes=opt.score_function_num_processes,
                                                **scoring_function_kwargs)
        self.scores = self.scoring_function(self.pred_sents)
        # self.scores = self.scoring_function([self.src_raw_without_L]*len(pred_sents), [self.gold_sent]*len(pred_sents), self.pred_sents)

    def log(self):
        """
        Log translation.
        """

        output = '\nSENT: {}\n'.format(self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.agent_scores[0]
        output += 'PRED: {}\n'.format(best_pred)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            output += 'GOLD: {}\n'.format(self.gold_sent)
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for agent_score, sent, prior_score, score in zip(self.agent_scores, self.pred_sents, self.prior_scores, self.scores):
                output += "[agent likelihood {:.4f}] [prior likelihood {:.4f}] [score {:.2f}] {}\n".format(
                    agent_score, prior_score, score, sent)



        return output
