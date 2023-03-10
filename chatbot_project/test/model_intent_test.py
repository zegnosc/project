# 경로 추가
import sys
sys.path.append(".")

from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic = "train_tools/dict/chatbot_dict.bin")

intent = IntentModel(model_name = "models/intent/intent_model.h5", preprocess = p)

query = "탕수육을 1시간 이내에 가져다 주지 않으면 지구를 파괴하겠다"

predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)