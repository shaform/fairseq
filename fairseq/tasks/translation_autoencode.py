# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import options
from fairseq.tasks.translation import TranslationTask

from . import register_task


@register_task("translation_autoencode")
class TranslationAutoEncodeTask(TranslationTask):
    """
    TranslationAutoEncode
    """

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        left_pad_source = options.eval_bool(args.left_pad_source)
        assert not left_pad_source, 'must pad source on the right!'
        return super(TranslationAutoEncodeTask, cls).setup_task(args, **kwargs)
