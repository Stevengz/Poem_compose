import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
import numpy as np
import os

# 导入数据集
data = np.load('data.npy', allow_pickle=True).tolist()
data_line = np.array([word for poem in data for word in poem])
ix2word = np.load('ix2word.npy', allow_pickle=True).item()
word2ix = np.load('word2ix.npy', allow_pickle=True).item()


# 构建模型的函数
def GRU_model(vocab_size, embedding_dim, units, batch_size):
    model = keras.Sequential([
        layers.Embedding(vocab_size,
                         embedding_dim,
                         batch_input_shape=[batch_size, None]),
        layers.GRU(units,
                   return_sequences=True,
                   stateful=True,
                   recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    return model

# 切分成输入和输出
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# 损失函数
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits,
                                                           from_logits=True)


# 每批大小
BATCH_SIZE = 64
# 缓冲区大小
BUFFER_SIZE = 10000
# 训练周期
EPOCHS = 20
# 诗的长度
poem_size = 125
# 嵌入的维度
embedding_dim = 64
# RNN 的单元数量
units = 128

# 创建训练样本
poem_dataset = tf.data.Dataset.from_tensor_slices(data_line)
# 将每首诗提取出来并切分成输入输出
poems = poem_dataset.batch(poem_size + 1, drop_remainder=True)
dataset = poems.map(split_input_target)
# 分批并随机打乱
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 创建模型
model = GRU_model(vocab_size=len(ix2word),
                  embedding_dim=embedding_dim,
                  units=units,
                  batch_size=BATCH_SIZE)
model.summary()
model.compile(optimizer='adam', loss=loss)
# 检查点目录
checkpoint_dir = './training_checkpoints'
# 检查点设置
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True)

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
