import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def GRU_model(vocab_size, embedding_dim, units, batch_size):
    model = keras.Sequential([
        layers.Embedding(vocab_size,
                         embedding_dim,
                         batch_input_shape=[batch_size, None]),
        layers.GRU(units,
                   return_sequences=True,
                   stateful=True,
                   recurrent_initializer='glorot_uniform'),
        # layers.GRU(units),
        layers.Dense(vocab_size)
    ])
    return model


# poem_type 为 0 表示整体生成，为 1 表示生成藏头诗
def generate_text(model, start_string, poem_type):
    # 控制诗句意境
    prefix_words = '月上柳梢头，人约黄昏后。'
    # 要生成的字符个数
    num_generate = 120
    # 空字符串用于存储结果
    poem_generated = []
    temperature = 1.0
    # 以开头正常生成
    if poem_type == 0:
        # 将整个输入直接导入
        input_eval = [word2ix[s] for s in prefix_words + start_string]
        # 添加开始标识
        input_eval.insert(0, word2ix['<START>'])
        input_eval = tf.expand_dims(input_eval, 0)
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # 删除批次的维度
            predictions = tf.squeeze(predictions, 0)
            # 用分类分布预测模型返回的字符
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions,
                                                 num_samples=1)[-1, 0].numpy()
            # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
            input_eval = tf.expand_dims([predicted_id], 0)
            poem_generated.append(ix2word[predicted_id])
        # 删除多余的字
        del poem_generated[poem_generated.index('<EOP>'):]
        return (start_string + ''.join(poem_generated))
    # 藏头诗
    if poem_type == 1:
        for i in range(len(start_string)):
            # 藏头诗以每个字分别生成诗句
            input_eval = [word2ix[s] for s in prefix_words + start_string[i]]
            input_eval.insert(0, word2ix['<START>'])
            input_eval = tf.expand_dims(input_eval, 0)
            model.reset_states()
            poem_one = [start_string[i]]
            for j in range(num_generate):
                predictions = model(input_eval)
                # 删除批次的维度
                predictions = tf.squeeze(predictions, 0)
                # 用分类分布预测模型返回的字符
                predictions = predictions / temperature
                predicted_id = tf.random.categorical(
                    predictions, num_samples=1)[-1, 0].numpy()
                # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
                input_eval = tf.expand_dims([predicted_id], 0)
                poem_one.append(ix2word[predicted_id])
            # 删除多余的字
            del poem_one[poem_one.index('。') + 1:]
            poem_generated.append(''.join(poem_one) + '\n')
        return ''.join(poem_generated)


ix2word = np.load('ix2word.npy', allow_pickle=True).item()
word2ix = np.load('word2ix.npy', allow_pickle=True).item()
# 嵌入的维度
embedding_dim = 64
# RNN 的单元数量
units = 128

model = GRU_model(len(ix2word), embedding_dim, units=units, batch_size=1)

checkpoint_dir = './training_checkpoints'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

print(generate_text(model, start_string="深度学习", poem_type=1))