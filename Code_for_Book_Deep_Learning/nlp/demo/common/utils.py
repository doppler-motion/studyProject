import numpy as np
import os, sys

from sklearn.utils.extmath import randomized_svd


from .data import load_data
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.log_config import logger



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def create_to_matrix(corpus, vocab_size, windows_size=1):
    """
    生成共现矩阵
    :param corpus: 单词id列表
    :param vocab_size: 单词个数
    :param windows_size: 窗口大小
    :return:
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, windows_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# eps是为了防止全为0的数据
def cos_similarity(x, y, eps=1e-8):
    """余弦相似度"""
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)  # x 的正则化
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)  # y 的正则化
    return np.dot(nx, ny)


# 返回某个查询词相似度前几的单词
def most_similarity(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    找出与单词 相似度最高的几个
    :param query: 要查找的单词
    :param word_to_id: 单词到单词id的字典
    :param id_to_word: 单词ID到单词的字典
    :param word_matrix: 共现矩阵
    :param top: 个数
    :return:
    """
    if query not in word_to_id:
        print("%s is not found" % query)
        return

    print("[query]", query)

    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 计算余弦相似度
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, word_matrix[i])

    # 按降序输出相似度
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue

        print(f"word: {id_to_word[i]}, similarity: {similarity[i]}")
        count += 1
        if count > top:
            return


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


def ppmi(C, verbose=False, eps=1e-8):
    """
    正的点互信息，便是两个单词的相关性
    :param C: 共现矩阵
    :param verbose: 是否输出计算过程
    :param eps: 微小值，用于防止出现无穷大值得情况
    :return:
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)

    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print("%.1f%% done." % (100 * cnt / total))

    return M


def create_contexts_targets(corpus, windows_size=1):
    """
    生成上下文和目标词
    :param corpus: 单词id列表
    :param windows_size: 窗口大小
    :return:
    """
    target = corpus[windows_size:-windows_size]
    contexts = []

    for idx in range(windows_size, len(corpus) - windows_size):
        cs = []
        for t in range(-windows_size, windows_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


if __name__ == "__main__":
    # word_text = "Python is a very great language and can achieve many funcs."
    # corpus, word_to_id, id_to_word = process(word_text)
    #
    # vocab_size = len(word_to_id)
    # co_matrix = create_to_matrix(corpus, vocab_size=vocab_size)
    # logger.info(f"co_matrix: {co_matrix}")
    # W = ppmi(co_matrix)
    # logger.info("W 正点互信息: ")
    # logger.info(W)
    # U, V , D = np.linalg.svd(W)

    # 基于ptb语料库的学习
    corpus, word_to_id, id_to_word = load_data("train")
    logger.info(f"corpus size: {len(corpus)}")
    logger.info(f"corpus[:30]: {corpus[:30]}")
    logger.info("")
    logger.info(f"id_to_word[0]: {id_to_word[0]}")
    logger.info(f"id_to_word[1]: {id_to_word[1]}")
    logger.info(f"id_to_word[2]: {id_to_word[2]}")
    logger.info("")
    logger.info(f"word_to_id['car']: {word_to_id['car']}")
    logger.info(f"word_to_id['happy']: {word_to_id['happy']}")
    logger.info(f"word_to_id['lexus']: {word_to_id['lexus']}")

    window_size = 2
    wordvec_size = 100
    vocab_size = len(word_to_id)

    logger.info("counting co-occurrence ...")
    co_matrix = create_to_matrix(corpus=corpus, windows_size=window_size, vocab_size=vocab_size)
    logger.info(f"co_matrix : {co_matrix.shape}")
    logger.info("calcuating PPMI...")
    W = ppmi(co_matrix, verbose=True)

    try:
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        U, S, V = np.linalg.svd(W)

    word_vec = U[:, :wordvec_size]

    query = ["you", "car", "game", "year"]
    for item in query:
        most_similarity(item, word_to_id, id_to_word, word_vec, top=5)

