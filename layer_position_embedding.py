# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 14:04
# @Author  : Yu Pengfei
# @Email   : 1837580905@qq.com
# @File    : layer_position_embedding.py
# @Software: PyCharm
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate


# 按位置计算位置嵌入比例
def keep_normal_distribution(max_len, ratio):
    # 参数ratio，一个超参， 可以理解为位置峰值附近的方面词嵌入比例
    res = []
    for i in range(max_len):
        x = - (max_len - 1 - i) ** 2 / (2 * ratio ** 2)
        r = tf.exp(x)
        res.append(r)
    res = tf.constant(tf.cast(res, dtype=tf.float32))
    res_n = []
    for i in range(max_len):
        x = - i ** 2 / (2 * ratio ** 2)
        r = tf.exp(x)
        res_n.append(r)
    res_n = tf.constant(tf.cast(res_n, dtype=tf.float32))
    return res, res_n


# 计算位置嵌入加权权重
def calculate_location_weight(batch, max_len, peak_loc, weight, weight_n):
    w_list = []
    for i in range(batch):
        before = tf.slice(weight, [max_len-peak_loc[i][0]-1], [peak_loc[i][0]+1])
        after = tf.slice(weight_n, [1], [max_len-peak_loc[i][0]-1])
        pos_weight = tf.concat([before, after], 0)
        pos_weight = tf.expand_dims(pos_weight, axis=1)
        pos_weight = tf.tile(pos_weight, (1, 768))
        pos_weight = tf.expand_dims(pos_weight, axis=0)
        w_list.append(pos_weight)
    pw = Concatenate(axis=0)(w_list)
    return pw


def position_embedding(bert: tf.Tensor, aspect: tf.Tensor, batch: int, max_length: int):
    s = tf.matmul(aspect, bert, transpose_b=True)
    p_max = tf.argmax(s, axis=2)
    r1, r2 = keep_normal_distribution(max_length, ratio=24)
    w1 = calculate_location_weight(batch, max_length, p_max, r1, r2)
    emb_bert = w1 * bert
    w2 = tf.ones([batch, max_length, 768])
    w2 = w2 - w1
    aspect_exp = tf.tile(aspect, (1, max_length, 1))
    emb_aspect = w2 * aspect_exp
    cnn_emb = emb_bert + emb_aspect

    return cnn_emb


# 计算位置嵌入加权权重
def calculate_location_weight_w(batch, max_len, peak_loc, weight, weight_n):
    w_list = []
    for i in range(batch):
        before = tf.slice(weight, [max_len-peak_loc[i][0]-1], [peak_loc[i][0]+1])
        after = tf.slice(weight_n, [1], [max_len-peak_loc[i][0]-1])
        pos_weight = tf.concat([before, after], 0)
        pos_weight = tf.expand_dims(pos_weight, axis=1)
        pos_weight = tf.tile(pos_weight, (1, 300))
        pos_weight = tf.expand_dims(pos_weight, axis=0)
        w_list.append(pos_weight)
    pw = Concatenate(axis=0)(w_list)
    return pw


def position_embedding_w(bert: tf.Tensor, aspect: tf.Tensor, batch: int, max_length: int):
    s = tf.matmul(aspect, bert, transpose_b=True)
    p_max = tf.argmax(s, axis=2)
    r1, r2 = keep_normal_distribution(max_length, ratio=8)
    w1 = calculate_location_weight_w(batch, max_length, p_max, r1, r2)
    emb_bert = w1 * bert
    w2 = tf.ones([batch, max_length, 300])
    w2 = w2 - w1
    aspect_exp = tf.tile(aspect, (1, max_length, 1))
    emb_aspect = w2 * aspect_exp
    cnn_emb = emb_bert + emb_aspect

    return cnn_emb



if __name__ == '__main__':
    r1, r2 = keep_normal_distribution(64, 6)
    pl = tf.random.uniform([4, 1], minval=0, maxval=63, dtype=tf.int32)
    calculate_location_weight(64, pl, r1, r2)