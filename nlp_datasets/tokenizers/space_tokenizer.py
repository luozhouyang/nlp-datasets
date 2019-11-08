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
