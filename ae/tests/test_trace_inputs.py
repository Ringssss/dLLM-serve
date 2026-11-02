import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace import make_input


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_surrogate_has_exact_lengths_even_shorter_than_template():
    tokenizer = Tokenizer()
    for n in [1, 3, 32, 128, 1024]:
        assert len(make_input(tokenizer, 'real prompt text', n)) == n
