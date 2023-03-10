# utils를 찾지 못 해 경로를 추가해 해결
import sys
sys.path.append(".")

# # 인터프리터 설정이 달라 에러가 발생할 경우 아나콘다 가상환경 모듈 라이브러리 경로를 추가해 해결하거나 실행 인터프리터를 변경해 해결
# sys.path.append("C:\Users\admin\anaconda3\envs\data\Lib\site-packages")

from utils.Preprocess import Preprocess

sent = "내일 오전 10시에 짬뽕 주문하고 싶어"

# 전처리 객체 생성
p = Preprocess()

# 형태소 분석기 실행
pos = p.pos(sent)

# 품사 태그와 같이 키워드 출력
ret = p.get_keywords(pos, without_tag = False)
print(ret)

# 품사 태그 없이 키워드 출력
ret = p.get_keywords(pos, without_tag = True)
print(ret)

# 결과

# [('내일', 'NNG'), ('오전', 'NNP'), ('10', 'SN'), ('시', 'NNB'), ('짬뽕', 'NNP'), ('주문', 'NNG'), ('싶', 'VX')]
# ['내일', '오전', '10', '시', '짬뽕', '주문', '싶']