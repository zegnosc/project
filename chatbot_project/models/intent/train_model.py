# 경로 추가
import sys
sys.path.append(".")
# sys.path.append("C:\jupyter_study\chatbot\chatbot_project")

import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
from utils.Preprocess import Preprocess
from config.GlobalParams import MAX_SEQ_LEN

# 데이터 읽어오기
data = pd.read_csv("models/intent/total_train_data.csv", delimiter = ",")

queries = data["query"].tolist()
intents = data["intent"].to_list()

p = Preprocess(word2index_dic = "train_tools/dict/chatbot_dict.bin")

# 단어 시퀀스 생성
sequences = []

for sentence in queries:
    pos = p.pos(sentence)
    keyword = p.get_keywords(pos, without_tag = True)
    seq = p.get_wordidx_sequence(keyword)
    sequences.append(seq)

# 패딩
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen = MAX_SEQ_LEN, padding = "post")

print(padded_seqs.shape)
print(len(intents))

# 학습 : 검증 : 테스트 = 7 : 2 : 1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
ds = ds.shuffle(len(queries))

train_size = int(len(padded_seqs) * .7)
val_size = int(len(padded_seqs) * .2)
test_size = int(len(padded_seqs) * .1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

# 하이퍼파라미터 설정
dropout_prob = .5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(p.word_index) + 1 # 전체 단어 수


# CNN 모델 정의
input_layer = Input(shape = (MAX_SEQ_LEN))

embedding_layer = Embedding(VOCAB_SIZE,
                            EMB_SIZE,
                            input_length = MAX_SEQ_LEN)(input_layer)

dropout_emb = Dropout(rate = dropout_prob)(embedding_layer)


conv1 = Conv1D(filters = 128,
               kernel_size = 3,
               padding = "valid",
               activation = tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)


conv2 = Conv1D(filters = 128,
               kernel_size = 4,
               padding = "valid",
               activation = tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)


conv3 = Conv1D(filters = 128,
               kernel_size = 5,
               padding = "valid",
               activation = tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3, 4, 5-gram 이 후 결합
concat = concatenate([pool1, pool2, pool3])


hidden = Dense(128, activation = tf.nn.relu)(concat)

dropout_hidden = Dropout(rate = dropout_prob)(hidden)

logits = Dense(5, name = "logits")(dropout_hidden)

predictions = Dense(5,
                    activation = tf.nn.softmax)(logits)

# 모델 생성
model = Model(inputs = input_layer,
              outputs = predictions)

model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = "accuracy")

# 모델 학습
# 모델 학습이 제대로 되지 않아 에포크 6으로 변경
model.fit(train_ds,
          validation_data = val_ds,
          epochs = 6, verbose = 1)

# 모델 평가
loss, accuracy = model.evaluate(test_ds, verbose = 1)
print(f"Accuracy : {accuracy * 100}")
print(f"Loss : {loss}")

# 모델 저장
model.save("models/intent/intent_model.h5")