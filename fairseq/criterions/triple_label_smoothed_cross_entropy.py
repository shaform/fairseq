# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion('triple_label_smoothed_cross_entropy')
class TripleLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss_src, nll_loss_src, loss_tgt, nll_loss_tgt, mse_loss = self.compute_loss(
                model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        mse_loss = mse_loss * sample_size / sample['target'].size(0)
        logging_output = {
            'loss_src': utils.item(loss_src.data) if reduce else loss_src.data,
            'nll_loss_src': utils.item(nll_loss_src.data) if reduce else nll_loss_src.data,
            'loss_tgt': utils.item(loss_tgt.data) if reduce else loss_tgt.data,
            'nll_loss_tgt': utils.item(nll_loss_tgt.data) if reduce else nll_loss_tgt.data,
            'mse_loss': utils.item(mse_loss.data) if reduce else mse_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        total_loss = loss_src + loss_tgt + mse_loss
        return total_loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs_src = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs_src = lprobs_src.view(-1, lprobs_src.size(-1))

        lprobs_tgt = model.get_normalized_probs(net_output[1], log_probs=True)
        lprobs_tgt = lprobs_tgt.view(-1, lprobs_tgt.size(-1))

        target = model.get_targets(sample, net_output).view(-1, 1)
        loss_src, nll_loss_src = label_smoothed_nll_loss(
            lprobs_src, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        loss_tgt, nll_loss_tgt = label_smoothed_nll_loss(
            lprobs_tgt, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        mse_loss = net_output[2]
        return loss_src, nll_loss_src, loss_tgt, nll_loss_tgt, mse_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_src_sum = sum(log.get('loss_src', 0) for log in logging_outputs)
        nll_loss_src_sum = sum(log.get('nll_loss_src', 0) for log in logging_outputs)
        loss_tgt_sum = sum(log.get('loss_tgt', 0) for log in logging_outputs)
        nll_loss_tgt_sum = sum(log.get('nll_loss_tgt', 0) for log in logging_outputs)
        mse_loss_sum = sum(log.get('mse_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss_src', loss_src_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss_src', nll_loss_src_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl_src', lambda meters: round(2**meters['nll_loss_src'].avg, 3))
        metrics.log_scalar('loss_tgt', loss_tgt_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss_tgt', nll_loss_tgt_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl_tgt', lambda meters: round(2**meters['nll_loss_tgt'].avg, 3))
        metrics.log_scalar('mse_loss', mse_loss_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
