from nlp_datasets.tokenizers import AbstractTokenizer


class SpaceTokenizer(AbstractTokenizer):
    """Tokenize strings by SPACE."""

    def __init__(self, config=None):
        super(SpaceTokenizer, self).__init__(config)

    def _process_line(self, line, token_filters=None):
        for w in line.split(' '):
            w = w.strip()
            if not w:
                continue
            if token_filters:
                if any(f.drop(w) for f in token_filters):
                    continue
            self.counter[w] = self.counter.get(w, 0) + 1


class XYSpaceTokenizer(SpaceTokenizer):

    def _process_line(self, line, token_filters=None):
        if not line:
            return
        seqs = line.split(self.config['xy_sep'])  # [x, y]
        for seq in seqs[0:2]:  # [x, y]
            super(XYSpaceTokenizer, self)._process_line(seq, token_filters)

    def _get_default_config(self):
        p = super(XYSpaceTokenizer, self)._get_default_config()
        p.update({
            "xy_sep": '@@@'
        })
        return p


class XYZSpaceTokenizer(SpaceTokenizer):
    def _process_line(self, line, token_filters=None):
        if not line:
            return
        seqs = line.split(self.config['xyz_sep'])  # [x, y, z]
        for seq in seqs[0:2]:  # [x, y]
            super(XYZSpaceTokenizer, self)._process_line(seq, token_filters)

    def _get_default_config(self):
        p = super(XYZSpaceTokenizer, self)._get_default_config()
        p.update({
            "xyz_sep": '@@@'
        })
        return p
