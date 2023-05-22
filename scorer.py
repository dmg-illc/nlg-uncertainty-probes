from functools import cache

import numpy as np
import spacy
import torch
from nltk import ngrams
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
import torch.nn.functional as F
from scipy.spatial.distance import cosine, euclidean
from typing import List


class Scorer:
    def __init__(self, lang: str):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.spacy_processor = (
            spacy.load("en_core_web_sm") if lang == "en-sent" else spacy.load("de_core_news_md")
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
            if lang == "en-sent"
            else AutoTokenizer.from_pretrained(
                "sentence-transformers/distiluse-base-multilingual-cased-v1"
            )
        )
        self.model = (
            AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")
            .eval()
            .to(self.device)
            if lang == "en-sent"
            else AutoModelWithLMHead.from_pretrained(
                "sentence-transformers/distiluse-base-multilingual-cased-v1"
            )
            .eval()
            .to(self.device)
        )
        self.lang = lang

    @cache
    def _tokenize(self, s: str, max_len: int = None):
        # Cut string at the LM sub-word token length to mimic the generation setting
        if max_len:
            s = self.tokenizer.decode(
                self.tokenizer(
                    s,
                    padding=False,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                ).input_ids
            )

        doc = self.spacy_processor(s.strip())
        return doc

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.hidden_states[-1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def ngram_overlap(
        self,
        string1: str,
        string2: str,
        max_len1: int,
        max_len2: int,
        n: int,
        pos: bool = False,
    ):
        tokenized_string1 = self._tokenize(string1, max_len=max_len1)
        ngrams1 = list(
            ngrams(
                [token.pos_ if pos else token.text.lower() for token in tokenized_string1],
                n,
            )
        )
        tokenized_string2 = self._tokenize(string2, max_len=max_len2)
        ngrams2 = list(
            ngrams(
                [token.pos_ if pos else token.text.lower() for token in tokenized_string2],
                n,
            )
        )

        count_1_in_2 = sum([1 if ngram2 in ngrams1 else 0 for ngram2 in ngrams2])
        count_2_in_1 = sum([1 if ngram1 in ngrams2 else 0 for ngram1 in ngrams1])
        combined_length = len(ngrams1) + len(ngrams2)
        return (
            (count_1_in_2 + count_2_in_1) / combined_length if combined_length > 0 else float("nan")
        )

    def compute_embeddings(self, strings: List[str], max_len: int):
        tokenized_strings = [self._tokenize(string, max_len=max_len).text for string in strings]
        encoded_input = self.tokenizer(tokenized_strings, padding=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.device), output_hidden_states=True)
        batch_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        return F.normalize(batch_embeddings, p=2, dim=1).to("cpu").numpy()

    @staticmethod
    def cosine_similarity(embed1: np.ndarray, embed2: np.ndarray):
        return -cosine(embed1, embed2) + 1

    @staticmethod
    def euclidean_similarity(embed1: np.ndarray, embed2: np.ndarray):
        return 1 / (1 + euclidean(embed1, embed2))

    def length(self, string: str):
        doc = self._tokenize(string)
        return len([token for token in doc])
