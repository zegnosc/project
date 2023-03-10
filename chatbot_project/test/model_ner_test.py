# 경로 추가
import sys
sys.path.append(".")
# sys.path.append("C:\jupyter_study\chatbot\chatbot_project")

from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic = "train_tools/dict/chatbot_dict.bin")

ner = NerModel(model_name = "models/ner/ner_model.h5", preprocess = p)

query = "오늘 오전 13시 2분에 탕수육 주문 하고 싶어요"

predicts = ner.predict(query)
tags  = ner.predict_tag(query)
print(predicts)
print(tags)