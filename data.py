import os
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
import pytorch_lightning as pl

from utils_t import save_pkl, load_pkl


class Multi30k(pl.LightningDataModule):

    def __init__(self,
                 lang=("en", "fr"),
                 max_seq_len=256,
                 unk_idx=0,
                 pad_idx=1,
                 sos_idx=2,
                 eos_idx=3,
                 vocab_min_freq=2):

        self.dataset_name = "multi30k"
        self.lang_src = "en"
        self.lang_tgt = "fr"
        self.max_seq_len = max_seq_len
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk = "<unk>"
        self.pad = "<pad>"
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.specials={
                self.unk: self.unk_idx,
                self.pad: self.pad_idx,
                self.sos: self.sos_idx,
                self.eos: self.eos_idx
                }
        self.vocab_min_freq = vocab_min_freq

        self.tokenizer_src = self.build_tokenizer(self.lang_src)
        self.tokenizer_tgt = self.build_tokenizer(self.lang_tgt)

        self.train = None
        self.valid = None
        self.test = None
        # self.build_dataset()

        self.vocab_src = None
        self.vocab_tgt = None
        # self.build_vocab()

        self.transform_src = None
        self.transform_tgt = None
        # self.build_transform()

        self.raw_dir = "raw"
        self.cache_dir = "/home/project/SITH/.data"
        self.batch_size = 1

    def prepare_data(self):
        cache_dir = os.path.join(self.cache_dir, self.dataset_name)
        raw_dir = os.path.join(cache_dir, self.raw_dir)
        os.makedirs(raw_dir, exist_ok=True)

        train_file = os.path.join(cache_dir, "train.pkl")
        valid_file = os.path.join(cache_dir, "valid.pkl")
        test_file = os.path.join(cache_dir, "test.pkl")

        if os.path.exists(train_file):
            self.train = load_pkl(train_file)
        else:
            with open(os.path.join(raw_dir, "train.en"), "r") as f:
                train_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "train.fr"), "r") as f:
                train_fr = [text.rstrip() for text in f]
            self.train = [(en, fr) for en, fr in zip(train_en, train_fr)]
            save_pkl(self.train , train_file)

        if os.path.exists(valid_file):
            self.valid = load_pkl(valid_file)
        else:
            with open(os.path.join(raw_dir, "val.en"), "r") as f:
                valid_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "val.fr"), "r") as f:
                valid_fr = [text.rstrip() for text in f]
            self.valid = [(en, fr) for en, fr in zip(valid_en, valid_fr)]
            save_pkl(self.valid, valid_file)

        if os.path.exists(test_file):
            self.test = load_pkl(test_file)
        else:
            with open(os.path.join(raw_dir, "test_2016_flickr.en"), "r") as f:
                test_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "test_2016_flickr.fr"), "r") as f:
                test_fr = [text.rstrip() for text in f]
            self.test = [(en, fr) for en, fr in zip(test_en, test_fr)]
            save_pkl(self.test, test_file)
        self.build_vocab()
        self.build_transform()

    def setup(self):
        pass


    def build_vocab(self, cache_dir=".data"):
        assert self.train is not None
        def yield_tokens(is_src=True):
            for text_pair in self.train:
                if is_src:
                    yield [str(token) for token in self.tokenizer_src(text_pair[0])]
                else:
                    yield [str(token) for token in self.tokenizer_tgt(text_pair[1])]

        cache_dir = os.path.join(cache_dir, self.dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

        vocab_src_file = os.path.join(cache_dir, f"vocab_{self.lang_src}.pkl")
        if os.path.exists(vocab_src_file):
            vocab_src = load_pkl(vocab_src_file)
        else:
            vocab_src = build_vocab_from_iterator(yield_tokens(is_src=True), min_freq=self.vocab_min_freq, specials=self.specials.keys())
            vocab_src.set_default_index(self.unk_idx)
            save_pkl(vocab_src, vocab_src_file)

        vocab_tgt_file = os.path.join(cache_dir, f"vocab_{self.lang_tgt}.pkl")
        if os.path.exists(vocab_tgt_file):
            vocab_tgt = load_pkl(vocab_tgt_file)
        else:
            vocab_tgt = build_vocab_from_iterator(yield_tokens(is_src=False), min_freq=self.vocab_min_freq, specials=self.specials.keys())
            vocab_tgt.set_default_index(self.unk_idx)
            save_pkl(vocab_tgt, vocab_tgt_file)

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt


    def build_tokenizer(self, lang):
        from torchtext.data.utils import get_tokenizer
        spacy_lang_dict = {
                'en': "en_core_web_sm",
                'fr': "fr_core_news_sm"
                }
        assert lang in spacy_lang_dict.keys()
        return get_tokenizer("spacy", spacy_lang_dict[lang])


    def build_transform(self):
        def get_transform(self, vocab):
            return T.Sequential(
                    T.VocabTransform(vocab),
                    T.Truncate(self.max_seq_len-2),
                    T.AddToken(token=self.sos_idx, begin=True),
                    T.AddToken(token=self.eos_idx, begin=False),
                    T.ToTensor(padding_value=self.pad_idx))

        self.transform_src = get_transform(self, self.vocab_src)
        self.transform_tgt = get_transform(self, self.vocab_tgt)


    def collate_fn(self, pairs):
        src = [self.tokenizer_src(pair[0]) for pair in pairs]
        tgt = [self.tokenizer_tgt(pair[1]) for pair in pairs]
        batch_src = self.transform_src(src)
        batch_tgt = self.transform_tgt(tgt)
        return (batch_src, batch_tgt)


    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, collate_fn=self.collate_fn)


    def translate(self, model, src_sentence: str, decode_func):
        model.eval()
        src = self.transform_src([self.tokenizer_src(src_sentence)]).view(1, -1)
        num_tokens = src.shape[1]
        tgt_tokens = decode_func(model,
                                 src,
                                 max_len=num_tokens+5,
                                 start_symbol=self.sos_idx,
                                 end_symbol=self.eos_idx).flatten().cpu().numpy()
        tgt_sentence = " ".join(self.vocab_tgt.lookup_tokens(tgt_tokens))
        return tgt_sentence

    def transex(self, model, src_sentence: str, decode_func):
        model.eval()
        src = self.transform_src([self.tokenizer_src(src_sentence)]).view(1, -1)
        num_tokens = src.shape[1]
        srcembedding, encoderall, decoderall = decode_func(model,
                                 src,
                                 max_len=num_tokens+5,
                                 start_symbol=self.sos_idx,
                                 end_symbol=self.eos_idx)
        # tgt_sentence = " ".join(self.vocab_tgt.lookup_tokens(tgt_tokens))
        return srcembedding, encoderall, decoderall

    def get_tgt_embed(self, model, tgt_sentence: str, decode_func):
        model.eval()
        tgt = self.transform_tgt([self.tokenizer_tgt(tgt_sentence)]).view(1, -1)
        tgt_embed = decode_func(model, tgt)
        return tgt_embed


    def getsrc(self, src_sentence: str):
        src = self.transform_src([self.tokenizer_src(src_sentence)]).view(1, -1)
        return src

if __name__=="__main__":
    ds = Multi30k()
    ds.prepare_data()
    ds.setup()
    print(ds.getsrc('A man in an orange hat starring at something.'))
    for src, tgt in ds.test_dataloader():
        print('输出src-------------------------------------------------------------')
        print(src)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        print(tgt_x)
        print(tgt_y)
        break
