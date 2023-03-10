# 경로 추가
import sys
sys.path.append(".")
# sys.path.append("C:\jupyter_study\chatbot\chatbot_project")

from config.DatabaseConfig import *
from utils.Database import Database
from utils.Preprocess import Preprocess
from utils.FindAnswer import FindAnswer
from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel


# 전처리 객체 생성
p = Preprocess(word2index_dic = "train_tools/dict/chatbot_dict.bin")

# 질문 / 답변 학습 디비 연결 객체 생성
db = Database(host = DB_HOST, user = DB_USER, password = DB_PASSWARD, db_name = DB_NAME)
# db 연결
db.connect()

# 원문
query = "거기 내 자리 사진 곤란"

# 의도 파악
intent = IntentModel(model_name = "models/intent/intent_model.h5", preprocess = p)
predict = intent.predict_class(query)
intent_name = intent.labels[predict]

# 개체명 인식
ner = NerModel(model_name = "models/ner/ner_model.h5", preprocess = p)
predicts = ner.predict(query)
ner_tags = ner.predict_tag(query)

print("질문 : ", query)
print("=" * 100)
print("의도 파악 : ", intent_name)
print("개체명 인식 : ", predicts)
print("답변 검색에 필요한 NER 태그 : ", ner_tags)
print("=" * 100)

try:
    f = FindAnswer(db)
    answer_text, answer_image = f.search(intent_name, ner_tags)
    answer = f.tag_to_word(predicts, answer_text)

except:
    answer = "죄송해요 무슨 말인지 모르겠어요"

print("답변 : ", answer)

# db 연결 해제
db.close()


# 의도 = 욕설
# 실제 = 주문
# 의도 분류 모델이 제대로 학습되지 않아 에포크 값 변경 후 모델 재생성