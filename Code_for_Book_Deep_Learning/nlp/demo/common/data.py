import os

import pickle
import numpy as np

dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/ptb"  # 被引用时

key_file = {
    'train': 'ptb.train.txt',
    'test': 'ptb.test.txt',
    'valid': 'ptb.valid.txt'
}
save_file = {
    'train': 'ptb.train.npy',
    'test': 'ptb.test.npy',
    'valid': 'ptb.valid.npy'
}
vocab_file = 'ptb.vocab.pkl'


def load_vocab_data():
    vocab_file_path = os.path.join(dataset_dir, vocab_file).replace("\\", "/")
    print(vocab_file_path)

    if os.path.exists(vocab_file_path):
        with open(vocab_file_path, "rb") as f:
            word_to_id, id_to_word = pickle.load(f)

        return word_to_id, id_to_word


def load_data(data_type="train"):
    if data_type == "val":
        data_type = "valid"

    word_to_id, id_to_word = load_vocab_data()
    corpus = {}

    save_path = dataset_dir + "/" + save_file[data_type]
    if os.path.exists(save_path):
        corpus = np.load(save_path)

    return corpus, word_to_id, id_to_word


if __name__ == "__main__":
    corpus, word_to_id, id_to_word = load_data()
    print("corpus size: %d" % len(corpus))
    print("word_to_id size: %d" % len(word_to_id))
    print("id_to_word size: %d" % len(id_to_word))
