from nlp_datasets.tokenizers import AbstractTokenizer


class SpaceTokenizer(AbstractTokenizer):
    """Tokenize strings by SPACE."""

    def __init__(self, config=None):
        super(SpaceTokenizer, self).__init__(config)
        self._index = self._vocab_size_include_unk
        self._special_tokens = [
            self.config['sos_token'],
            self.config['eos_token'],
            self.config['cls_token'],
            self.config['sep_token'],
            self.config['mask_token']
        ]

    def _process_line(self, line, token_filters=None):
        for w in line.split(' '):
            if w in self._special_tokens:
                continue
            if w in self._token2id_dict:
                continue
            if token_filters:
                if any(f.drop_token(w) for f in token_filters):
                    continue
            self._token2id_dict[w] = self._index
            self._id2token_dict[self._index] = w
            self._index += 1


class XYSpaceTokenizer(SpaceTokenizer):

    def _process_line(self, line, token_filters=None):
        if not line:
            return
        seqs = line.split(self.config['line_sep'])  # [x, y] or [x, y, z]
        for seq in seqs[0:2]:  # [x, y]
            for w in seq.split(' '):
                if w in self._special_tokens:
                    continue
                if w in self._token2id_dict:
                    continue
                if token_filters:
                    if any(f.drop_token(w) for f in token_filters):
                        continue
                self._token2id_dict[w] = self._index
                self._id2token_dict[self._index] = w
                self._index += 1

    def _get_default_config(self):
        p = super(XYSpaceTokenizer, self)._get_default_config()
        p.update({
            "line_sep": '@@@'
        })
        return p


class XYZSpaceTokenizer(XYSpaceTokenizer):
    pass
