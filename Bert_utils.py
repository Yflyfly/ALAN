# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 17:10
# @Author  : Yu Pengfei
# @Email   : 1837580905@qq.com
# @File    : Bert_utils.py
# @Software: PyCharm
import numpy
import pandas as pd
from transformers import BertTokenizer


def convert_example_to_feature(review, tokenizer, max_length):
    # 返回字典
    # {'input_ids': [101, 3284, 3449,......, 0, 0],
    #  'token_type_ids': [0, 0,...... 0, 0, 0],
    #  'attention_mask': [1, 1, 1,...... 0, 0, 0]}
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True,
                                 # return_offsets_mapping=True
                                 )


# 映射roberta模型的输入
def map_example_to_dict(input_ids, attention_masks, label):
    return {
               "input_ids": input_ids,
               "attention_mask": attention_masks,
           }, label


def map_emb_to_dict(input_ids, attention_masks, target_word, label):
    return {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "target_word": target_word
    }, label


def read_dataset(path):
    dataset = pd.read_csv(path, keep_default_na=False)
    return dataset


# 加载词向量并复制长度，返回numpy数组
def load_word_vector(path, copy_length):
    # import tensorflow as tf
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # （代表仅使用第0，1号GPU）
    emb = numpy.load(path)
    # tile() (1, )横向复制， ( ,1)纵向复制
    # emb = numpy.tile(emb, (copy_length, 1))
    # emb = tf.reshape(emb, (copy_length, 768))
    return emb


if __name__ == '__main__':
    # e = load_word_vector('data/embedding/bert_embed_env_w.npy', 32)
    # print(type(e), e.shape)
    # a = numpy.array([[[1, 2]], [[3, 4]]])
    # x = numpy.tile(a, (32, 1))
    # print(a, a.shape)
    # print(x, x.shape)
    tokenizer = BertTokenizer.from_pretrained('../file_PTMs/bert-base-uncased')

    s = 'prices are higher to dine in and their chicken tikka marsala is quite good'
    # string1 = "[CLS]" + s + "[SEP]"
    # tokens = convert_example_to_feature(s, tokenizer, 20)
    tokens = tokenizer.tokenize(s)
    # tokens['input_ids'] = tokens['input_ids'][1:]
    print(tokens)
    # [101, 2833, 2003, 2307, 102, 0, 0, 0, 0, 0]
