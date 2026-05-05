import re, collections

class bpe_tokenizer:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.merge_rules = []
        self.vocab = {}

    def train(self, corpus):
        """从原始文本语料训练，建立初始字符词表并学习 merge rules"""
        # 统计词频，初始化为字符序列
        word_freq = collections.defaultdict(int)
        for word in corpus.split():
            word_freq[word] += 1

        self.vocab = {
            ' '.join(list(word)) + ' </w>': freq
            for word, freq in word_freq.items()
        }

        # 训练循环
        for i in range(self.num_merges):
            pairs = self._get_stats()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self._merge_vocab(best)
            self.merge_rules.append(best)

        return self

    def encode(self, word):
        """把一个词编码为 subword token 列表"""
        symbols = list(word) + ['</w>']
        for rule in self.merge_rules:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == rule[0] and symbols[i+1] == rule[1]:
                    symbols = symbols[:i] + [''.join(rule)] + symbols[i+2:]
                else:
                    i += 1
        return symbols

    def tokenize(self, text):
        """对整段文本分词，返回所有 token"""
        tokens = []
        for word in text.split():
            tokens.extend(self.encode(word))
        return tokens

    def _get_stats(self):
        pairs = collections.defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def _merge_vocab(self, pair):
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in self.vocab:
            w_out = p.sub(''.join(pair), word)
            new_vocab[w_out] = self.vocab[word]
        self.vocab = new_vocab