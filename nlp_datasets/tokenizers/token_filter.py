import abc
import re

NUMBERS_PATTERN = re.compile('^[0-9]+$')


class TokenFilter(abc.ABC):

    def drop(self, token):
        raise NotImplementedError()


class EmptyTokenFilter(TokenFilter):

    def drop(self, token):
        if not token or not token.strip():
            return True
        return False


class NumbersTokenFilter(TokenFilter):

    def drop(self, token):
        m = NUMBERS_PATTERN.match(token)
        if m:
            return True
        return False


class LengthTokenFilter(TokenFilter):

    def __init__(self, max_len):
        self.max_len = max(0, max_len)

    def drop(self, token):
        return len(token) > self.max_len


class RegexTokenFilter(TokenFilter):

    def __init__(self, regex):
        self.pattern = re.compile(regex)

    def drop(self, token):
        m = self.pattern.match(token)
        if m:
            return True
        return False
