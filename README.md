# 프로젝트 모음 폴더

[](http://naver.me/5G56nfFF)

# [넷플릭스 데이터 분석](https://github.com/zegnosc/project/tree/main/Netflix_Project)

## 1. 프로젝트 소개
- 코로나19로 인해 전세계적으로 OTT서비스가 유행하는 가운데 가장 유명한 플랫폼인 넷플릭스의 컨텐츠에 대해 분석하고자 함 
- kaggle 사이트에 있는 IMDB, TMDB 사이트의 컨텐츠 정보 데이터 사용

## 2. 사용 기술, 라이브러리
- Python, Jupyter Notebook 사용
- pandas, numpy, matplotlib, seaborn, plotly

## 3. 분석 내용
- TV 프로그램과 영화 컨텐츠의 비율
    - 연도
    - 국가
    - 연령    
- 카테고리 데이터
    - 각 카테고리의 상관관계
    - IMDB, TMDB 사이트 점수
    
## 4. 결론
- 데이터 분석 프로젝트를 계획했으나 1차적인 분석이 되고 결론적으로 시각화 프로젝트로 마무리가 됨
- 문제점
    - 프로젝트 계획, 구조, 분석 방향 등 전체적인 계획 수립이 되지않아 부족한 점이 매우 많음
    - 분석 목표가 명확하지 않음
    - 데이터로 인해 얻을 수 있는 정보가 부족

## 5. 개선점
- 프로젝트 시작 시 전체적인 계획 수립 후 협업 툴을 사용해 일정 관리
    - 예) Notion, Trello, Flow 등
- 데이터 분석 관련 통찰을 넓히기 위해 추가적인 공부가 필요
    - EDA를 통한 데이터 분석 과정
- 분석을 토대로 실제 활용이 가능한 방향 고려
    - 예) 넷플릭스의 컨텐츠의 트렌드, 인프라 등 분석하고 조직내 의사결정에 활용이 가능한 데이터 가치 창출

## 참고 자료
- [데이터셋1](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies/versions/1?resource=download&select=titles.csv)
- [데이터셋2](https://www.kaggle.com/code/sukhdeepk/netflix-dataset-eda/data?select=netflix_titles.csv)

- - -

# [경제와 코로나 데이터를 통한 코로나19 동향 예측](https://github.com/zegnosc/project/tree/main/DLP_Project)

## 1. 프로젝트 소개
- 2020년 부터 코로나 확진자가 지속적으로 증가함에 따라 국가 방역 대책을 실시했으나 사회적 거리두기에 대한 반발, 백신접종에 대한 불신, 변종 바이러스로 위기가 있었지만 이 후 백신접종, 사회적 거리두기 등으로 과거에 비해 코로나 확진자 수는 많지만 완화된 방역 수칙으로 생활이 가능하게 됨
- 미래 다른 전염성 유행병 발병시 코로나 경험을 통해 방역 대책이나 다양한 우발 상황에 따른 확진자 수의 변화가 코로나와 유사하게 진행 될 것 이라고 생각 됨
- "시계열 데이터를 활용한 코로나19 동향 예측" 논문의 시계열 데이터를 통한 확진자 예측 모델을 접했고 scikit-learn의 모델들을 활용해 설명력이 높은 예측 모델을 만들어 미래 전염성 유행병에 대응할 수 있도록하고 현 시점 완화된 방역으로 인해 안일하게 생각할 수 있는 코로나를 강조
- 시계열 분석 같은 경우 시계열 데이터의 특징(추세, 계절, 순환, 불규칙)으로 예측하는데 코로나 데이터가 충분하지 않은 것 으로 판단해 회귀 분석으로 진행
- [Trello](https://trello.com/b/skmAgLSc/dlp%ED%8C%80)를 사용해 전체적인 진행 일정, 상황 공유

## 2. 사용기술, 라이브러리
- Python, Jupyter Notebook
- pandas, numpy, matplotlib, seaborn, plotly, statsmodels, scikit-learn, xgboost, lightgbm, catboost

## 3. 분석 내용
### 3-1. 기존 연구
- 시계열 데이터를 Prophet 알고리즘을 사용해 코로나19 동향 예측
  - 1년 3개월의 시계열 데이터를 통해 코로나 확진자를 예측.
  - Prophet 모델은 주기적이지 않은 변화인 Trene, Weekly, Yearly등, 주기적으로 나타나는 패턴 Seasonality, 휴일과 같이 불규칙한 이벤트로 성정하는 Holiday 등의 요소로 구성
  - <b>0.914</b>의 결정 계수를 가짐
-  코로나 사태가 국내 경제에 미치는 영향과 향후 과제
  - 코로나 사태로 국내 경제가 크게 둔화되고 있으며 지역 경제의 위기를 설명
  - 초기 이동제한조치, 국경봉쇄 조치에 따라 소비 감소, 노동시장 위축, 세계 무역규모 축소 등 경제에 직접적 영향을 미침
  - 온라인 등을 통한 비대면 산업은 빠른 성장을 이룸
  - 확진자, 사회적 거리두기 변화, 긴급 재난 지원금 등에 따라 국내 경제와 지역 경제에 변화가 있음을 설명
  
### 3-2. 실험 결과 및 분석
- 코로나 데이터에 경제 데이터를 추가해 다음 날 코로나 확진자를 예측하는 회귀분석 모델 생성
  - 여러 모델을 적용해 최적의 모델 선정
  - 평가 방법은 결정 계수, MAE, RMSE 등을 이용
- 코로나 데이터와 기업의 성장을 확인할 수 있는 코스피, 코스닥 컬럼과 환율, 소비자 물가 지수, 실업율 컬럼을 추가해 모델을 생성
- 코로나 이전의 경제 동향을 확인하기 위해 2018년 부터 데이터를 적용했으나 코로나 발생 전 까지의 모든 코로나 데이터 컬럼이 0 값으로 입력되어 모델 성능에 영향을 주고 결정 계수가 무의미하게 높게 나오는 것을 확인, 코로나 국내 발생 일자에 따라 2020-01-20 부터 시작으로 경제 데이터 적용
#### a. 코로나 데이터 회귀 분
- 최소 제곱법(OLS)
  - 2020-01-20 ~ 2022-12-31 기간의 코로나 확진자, 사망자, 백신 접종, 국가 통제(사회적 거리두기), 년, 월, 요일 컬럼을 통해 다음 날의 확진자 수를 예측하는 모델을 생성
  - 베이스라인 모델의 경우 target 값에 0이 많아 데이터의 분포가 정규분포 모형이 아니기 때문에 numpy의 log를 사용해 정규분포 모형에 가깝게 스케일링
  - 조건수가 100에 가까워 다중공선성 의심, 분산 팽창 요인이 10을 넘는 컬럼을 제외
  - <b>0.937</b>의 결정 계수를 가짐, 초기 모델에서는 <b>0.914</b>로 최종 모델보다 낮게 나왔지만 분산 팽창 요인이 높은 값을 제외해 다중공선성 위험을 줄임
- 의사결정나무
  - 평가 지표 MAE가 낮은 모델을 적용해 모델 생성
  - 기존 OLS모델 에서의 MAE를 확인하고 낮은 모델에 적용해 훈련 진행
  - <b>0.994</b>의 결정 계수와 <b>2599</b>의 MAE로 높은 성능을 보임
#### b. 코로나, 경제 데이터 회귀 분석
- 최소 제곱법(OLS)
  - 앞서 진행한 데이터와 동일하게 target 값에 log적용 후 데이터의 분포를 정규분포에 가깝게 스케일링
  - <b>0.940</b>의 결정 계수로 코로나 데이터만 적용된 모델보다 성능이 향상된 것으로 나왔으나 조건수가 148로 다중공선성이 의심, target 값과 독립변수들의 상관 관계가 높은 컬럼을 제외하고 p-value가 높은 컬럼을 제외해 최적화 진행
  - <b>0.938</b>의 결정 계수로 <b>0.002</b>의 수치가 하락했지만 조건수가 46.6으로 다중공선성 가능성을 낮춤
  - 선형 회귀 분석에서의 기본가정인 독립성을 따져 조건수가 가장 낮은 모델을 최종 모델로 선정
- 의사결정나무
  - 앞서 진행한 데이터와 동일하게 MAE가 가장 낮은 모델을 적용해 모델 생성
  - catboost를 사용한 모델이 <b>0.994</b>의 결정 계수와 <b>2379</b>의 MAE로 가장 높은 성능을 보임
  - scikit-learn 라이브러리의 RandomizedSearchCV와 optuna 라이브러리를 사용해 하이퍼 파라미터 튜닝을 시도했으나 MAE가 증가해 기존 모델을 그대로 적용

## 4. 결론
- 코로나 데이터와 경제 데이터를 적용해 다음 날 확진자를 예측하는 모델 생성, MAE(평균절대오차)를 기준으로 평가
- 미래 전염성 유행병을 방역 대책에 따른 변화를 예측하고 방역에 대한 중요성을 강조할 수 있을 것으로 판단
- 향후 연구로 병원 데이터와 집단 감염 사례를 포함한 데이터, 변종 바이러스 데이터 등이 포함된 데이터를 분석한다면 조금 더 좋은 성능의 데이터를 얻을 수 있음

## 5. 개선점
- 분석 자체가 코로나에 초점이 맞춰져 있어(국가 통제 단계) 타 질병에 적용한다면 설명력이 다소 부족할 수 있으므로 여러 케이스의 전염성 유행병 집단 감염 사례와 변종 바이러스의 병원데이터를 포함해 분석한다면 더 좋은 성능을 기대할 수 있을 것 으로 보임
  - 예) 메르스, 신종플루 등
- 분석 데이터를 기반으로 실제 서비스 활용이 가능한 방향 고려
  - 예) 확진자 예측을 통해 다음 날의 감염 위험도 표시와 방역 수칙 알림 등
- 분석 단계에서 모델을 생성했을 때 성능이 비교적 좋게 나와 OLS,  트리기반 모델을 사용했는데 시계열 데이터 예측에 성능이 좋은 RNN을 활용한 딥러닝 모델을 통해 예측하는 방향도 추가해 볼 수 있음

## 데이터
- [소비자 물가 지수](https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1J20003&vw_cd=MT_ZTITLE&list_id=P2_6&seqNo=&lang_mode=ko&language=kor&obj_var_id=&itm_id=&conn_path=MT_ZTITLE)
- [실업률](https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1DA7102S&vw_cd=MT_ZTITLE&list_id=B15&seqNo=&lang_mode=ko&language=kor&obj_var_id=&itm_id=&conn_path=MT_ZTITLE)
- [달러 원화 환율](https://finance.yahoo.com/quote/KRW%3DX/history?period1=1577836800&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)
- [코스피 지수](https://finance.yahoo.com/quote/%5EKS11/history?p=%5EKS11)
- [코스닥 지수](https://finance.yahoo.com/quote/%5EKQ11?p=^KQ11&.tsrc=fin-srch)
- [코로나 확진 / 사망자](https://ncov.kdca.go.kr/)
- [코로나 백신 1, 2차](https://ncv.kdca.go.kr/)
- [코로나 백신 3차](https://kdx.kr/data/view/30239)

## 참고 자료
- [시계열 데이터를 활용한 코로나19 동향 예측](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002741446)
- [다중 스태킹을 가진 새로운 앙상블 학습 기법](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002619294)
- [코로나-19 사태가 국내경제에 미치는 영향과 향후 과제](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002659150)
[](https://www.metroseoul.co.kr/article/20200915500473)
[](https://www.kukinews.com/newsView/kuk202008260315)
