import numpy as np

from loguru import logger


# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# define class labels
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

def process(text):
    """
    生成单词id列表、单词到单词id的字典、单词id到单词的字典
    :param text: 单词文本
    :return:
    """

    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    logger.info(f"word_to_id: {word_to_id}")
    logger.info(f"id_to_word: {id_to_word}")

    corpus = [word_to_id[w] for w in words]
    corpus = np.array(corpus)
    logger.info(f"corpus: {corpus}")

    return corpus, word_to_id, id_to_word


def convert_one_hot(corpus, vocab_size):
    """
    转成one-hot表示
    :param corpus: 单词id列表
    :param vocab_size: 词汇个数
    :return:
    """
    N = corpus.shape[0]
    one_hot = None

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx0, word_id1 in enumerate(corpus):
            for idx1, word_id2 in enumerate(word_id1):
                one_hot[idx0, idx1, word_id2] = 1
    return one_hot


def padding_sequences():
    pass


class EmbeddingLayer:
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # 初始化随机权重矩阵
        self.weights = np.random.randn(self.vocab_size, self.emb_dim) / np.sqrt(self.vocab_size)

    def forward(self, inputs):
        """
        :param inputs:  一组输入文本，每个文本由一个整数列表表示
        :return:  返回输入文本的嵌入向量表示，每个文本由一个形状为(emb_dim,)的向量表示
        """
        # 根据每个整数索引查找相应的向量
        embedded = self.weights[inputs]

        # 平均所有单词向量来表示整个文本
        return np.mean(embedded, axis=1)


if __name__ == "__main__":
    text = "The weather is not very nice today. Let's rest at home."
    corpus, word_to_id, id_to_word = process(text)
    logger.info("corpus: ", corpus)
    logger.info("word_to_id: ", word_to_id)
    logger.info("id_to_word: ", id_to_word)
