import IPython.display as ipd
import matplotlib.pyplot as plt

import torch
import torchaudio

from .data import train_set

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

# 寻找数据集中可用的标签列表
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

# 转换数据
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

ipd.Audio(transformed.numpy(), rate=new_sample_rate)


# 使用标签列表中的每个索引对每个单词进行编码
def label_to_index(word):  # 返回输入word的label索引下标
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):  # 返回索引对应的label
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


if __name__ == "__main__":
    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    # 查看波形
    # plt.plot(waveform.t().numpy())
    # plt.show()

    # # 35个标签是“命令”，前几个文件是marvin
    # waveform_first, *_ = train_set[0]
    # ipd.Audio(waveform_first.numpy(), rate=sample_rate)
    #
    # waveform_second, *_ = train_set[1]
    # ipd.Audio(waveform_second.numpy(), rate=sample_rate)
    #
    # # 最后一个文件是“视觉”
    # waveform_last, *_ = train_set[-1]
    # ipd.Audio(waveform_last.numpy(), rate=sample_rate)

    word_start = "yes"
    index = label_to_index(word_start)
    word_recovered = index_to_label(index)

    print(word_start, "-->", index, "-->", word_recovered)
