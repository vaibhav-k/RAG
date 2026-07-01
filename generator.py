"""
Text generation component.

This module loads the language model and tokenizer used to generate
answers from retrieved context. It receives a formatted prompt and
returns a natural language response.

The generator is independent of the retrieval pipeline, allowing
different language models to be substituted with minimal changes.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    GENERATOR_MODEL_NAME,
    MAX_NEW_TOKENS,
)


class Generator:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # do_sample=False,
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
