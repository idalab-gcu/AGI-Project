import re
import json
import pickle
import nltk
from collections import Counter

class Tokenizer:
    def __init__(self, config):
        self.ann_path = config.get("ann_path", "")
        self.threshold = config.get("threshold", 3)
        self.vocab_path = config.get("vocab_path", "vocab.pkl")
        
        self.pad_token = "<pad>"
        self.start_token = "<sos>"
        self.end_token = "<eos>"
        self.unk_token = "<unk>"
        self.special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        self._check_nltk()

    def _check_nltk(self):
        resources = ["punkt", "punkt_tab"]
        for res in resources:
            try:
                nltk.data.find(f"tokenizers/{res}")
            except LookupError:
                nltk.download(res, quiet=True)

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        return re.sub(r"\s+", " ", text).strip()

    def get_tokens(self, text):
        cleaned = self.clean_text(text)
        try:
            return nltk.word_tokenize(cleaned.lower())
        except:
            return cleaned.lower().split()

    def build_vocab(self):
        print(f" Building vocabulary from {self.ann_path}...")
        counter = Counter()
        with open(self.ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            report = item.get("report", item.get("text", ""))
            tokens = self.get_tokens(report)
            counter.update(tokens)

        for w in self.special_tokens:
            self.word2idx[w] = self.idx
            self.idx2word[self.idx] = w
            self.idx += 1

        for word, count in counter.most_common():
            if count >= self.threshold:
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1
        print(f" Vocabulary built! Total tokens: {len(self.word2idx)}")

    def __call__(self, text):
        tokens = self.get_tokens(text)
        ids = [self.word2idx[self.start_token]]
        for token in tokens:
            ids.append(self.word2idx.get(token, self.word2idx[self.unk_token]))
        ids.append(self.word2idx[self.end_token])
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if isinstance(i, list): i = i[0] # handle batch dimension quirk
            word = self.idx2word.get(int(i), self.unk_token)
            if word == self.end_token: break
            if word not in [self.pad_token, self.start_token, self.end_token]:
                tokens.append(word)
        return " ".join(tokens)

    def save_vocab(self):
        with open(self.vocab_path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)
        print(f" Vocabulary saved to {self.vocab_path}")

    def load_vocab(self):
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            self.word2idx = vocab['word2idx']
            self.idx2word = vocab['idx2word']
            self.idx = len(self.word2idx)
        print(f" Vocabulary loaded from {self.vocab_path} ({len(self.word2idx)} tokens)")