
from transformers import LogitsProcessor
import torch
from transformers import Constraint
import re
import TSED
LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


class ExtractiveLogitsProcessor(LogitsProcessor):
    def __init__(self, input_ids: torch.LongTensor, special_tokens: list):
        """
        Args:
            input_ids (torch.LongTensor): Tensor of shape (batch_size, sequence_length) representing input IDs.
            special_tokens (list): List of special token IDs that are always allowed.
        """
        self.allowed_tokens = set(input_ids.flatten().tolist()) | set(special_tokens)
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids (torch.LongTensor): Current input IDs (not used for filtering here).
            scores (torch.FloatTensor): Logits of shape (batch_size, vocab_size) to be processed.

        Returns:
            torch.FloatTensor: Processed logits with disallowed tokens set to `-inf`.
        """
   
        batch_size, vocab_size = scores.shape

        # Create a mask for allowed tokens
        allowed_mask = torch.full((vocab_size,), -float("inf"), device=scores.device)
        allowed_mask[list(self.allowed_tokens)] = 0  # Allowed tokens have logits unchanged

        # Expand the mask to match batch size
        allowed_mask = allowed_mask.unsqueeze(0).expand(batch_size, -1)  # (batch_size, vocab_size)
        scores = scores + allowed_mask
        return scores


class TSEDMonotonicConstraint(Constraint):
    def __init__(self, tokenizer, original_code, lang="java", eos_token_id=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.original_code = original_code
        self.lang = lang
        self.tsed = TSED
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

        self.generated_tokens = []
        self.prev_score = 0.0
        self._completed = False

    def advance(self):
        # Allow all tokens unless constraint has completed
        if self._completed:
            return None  # no constraint left
        return list(range(self.tokenizer.vocab_size))

    def does_advance(self, token_id: int):
        return not self._completed

    def update(self, token_id: int):
        self.generated_tokens.append(token_id)
        current_text = self.tokenizer.decode(self.generated_tokens, skip_special_tokens=False)
        cleaned = re.sub(r"<(backward|/backward|forward|/forward)>", "", current_text)

        score = self.tsed.Calculate(
            self.lang,
            self.original_code,
            remove_line_numbers(cleaned),
            1.0, 1.0, 1.0
        )

        if score < self.prev_score:
            self.reset()
            return False, False, True  # stepped, completed, reset

        self.prev_score = score

        if token_id == self.eos_token_id:
            self._completed = True
            return True, True, False  # stepped, completed, reset

        return True, False, False

    def reset(self):
        self.generated_tokens = []
        self.prev_score = 0.0
        self._completed = False

    def remaining(self):
        return 0 if self._completed else 1

    def copy(self, stateful=False):
        copied = TSEDMonotonicConstraint(self.tokenizer, self.original_code, self.tsed, self.lang, self.eos_token_id)
        if stateful:
            copied.generated_tokens = self.generated_tokens[:]
            copied.prev_score = self.prev_score
            copied._completed = self._completed
        return copied
