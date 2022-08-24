# -*- coding: utf-8 -*-
# @Time    : 2022/1/13 19:13
# @Author  : Yu Pengfei
# @Email   : 1837580905@qq.com
# @File    : model.py
# @Software: PyCharm
import os

import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Concatenate, LSTM
from transformers import BertTokenizer, TFBertModel
from tensorflow.python.keras.callbacks import ModelCheckpoint
from Bert_utils import convert_example_to_feature, read_dataset, map_emb_to_dict
from layer_position_embedding import position_embedding

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # （代表仅使用第0，1号GPU）

pre_model_path = 'bert-base-uncased'
train_path = '.data/r_train.csv'
test_path = '.data/r_test.csv'
filepath = "./model/best_model.h5"  # 模型保存路径
max_length = 64
type_nums = 3
batch_size = 16
learning_rate = 2e-5
epochs = 8

# 加载分词器和bert
tokenizer = BertTokenizer.from_pretrained(pre_model_path)
bert = TFBertModel.from_pretrained(pre_model_path)


# 生成aspect的bert词向量
def aspect_emb_gen(word: str):
    bert_input = tokenizer.encode(
                            word,
                            add_special_tokens=False,  # add [CLS], [SEP]
                            return_attention_mask=True,  # add attention mask to not focus on pad tokens
                  )
    inputs = tf.reshape(bert_input, [1, len(bert_input)])
    outputs = bert(inputs)
    return outputs[1]


food = aspect_emb_gen('food')
anecdotes = aspect_emb_gen('anecdotes / miscellaneous')
service = aspect_emb_gen('service')
ambience = aspect_emb_gen('ambience')
price = aspect_emb_gen('price')


# 构建输入数据集的例子
def encode_dataset(dataset, limit=-1):
    input_ids_list = []
    attention_mask_list = []
    vector_list = []
    label_list = []
    if limit > 0:
        dataset = dataset.take(limit)

    # 将数据集中每一行数据映射token
    for index, row in dataset.iterrows():
        review = row["text"]
        label = row["polarity_id"]
        target = row["category"]
        if target == "food":
            vector = food
        elif target == "anecdotes/miscellaneous":
            vector = anecdotes
        elif target == "service":
            vector = service
        elif target == "ambience":
            vector = ambience
        else:
            vector = price

        bert_input = convert_example_to_feature(review, tokenizer, max_length)
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        vector_list.append(vector)
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, vector_list, label_list)).map(
        map_emb_to_dict)


train_data = read_dataset(train_path)
test_data = read_dataset(test_path)
ds_train_encoded = encode_dataset(train_data).padded_batch(batch_size, drop_remainder=True)
ds_test_encoded = encode_dataset(test_data).padded_batch(batch_size, drop_remainder=True)


# 构建ALAN模型
class ALAN(object):
    def __init__(self, label_num):
        self.label_num = label_num

    def get_model(self):
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_masks = Input(shape=(None,), dtype=tf.int32, name="attention_masks")
        target_word = Input(shape=(None, 768), dtype=tf.float32, name='target_word')
        # bert输出最后一层隐向量
        outputs = bert(input_ids, attention_mask=attention_masks)
        # cls_output = outputs[1]
        outputs = outputs[0]

        # 计算内积作为相似度
        cnn_emb = position_embedding(outputs, target_word, batch_size, max_length)
        convs = []
        for kernel_size in [4, 5, 6]:
            c = Conv1D(256, kernel_size, padding="causal", activation='relu')(cnn_emb)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        p_cnn = Concatenate()(convs)

        # lstm = LSTM(units=768, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='LSTM')(cnn_emb)

        # attention
        # V = tf.keras.layers.Dense(1)
        # W1 = tf.keras.layers.Dense(300)
        # W2 = tf.keras.layers.Dense(300)
        # # score = V(tf.nn.tanh(tf.concat([W2(p_cnn_tile), W1(lstm)], 2)))
        # cl = tf.concat([W2(cnn_emb), W1(lstm)], 2)
        # score = V(tf.nn.tanh(cl))
        # score_zeros = -9e15 * tf.ones_like(score)
        # new_score = tf.where(tf.expand_dims(input_ids, axis=1) != 0, score, score_zeros)
        # attention_weights = tf.nn.softmax(new_score, axis=1)
        # context_vector = tf.matmul(attention_weights, lstm, transpose_a=True)
        # at_lstm = tf.reduce_mean(context_vector, axis=1)

        cla_outputs = Dense(self.label_num, activation='softmax')(p_cnn)
        model = Model(
            inputs={'input_ids': input_ids,
                    'attention_masks': attention_masks,
                    'target_word': target_word
                    },
            outputs=[cla_outputs])
        return model


model = ALAN(type_nums).get_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_weights_only=True,
                             save_best_only=True,
                             mode='max')

bert_history = model.fit(ds_train_encoded,
                         epochs=epochs,
                         validation_data=ds_test_encoded,
                         callbacks=[checkpoint]
                         )