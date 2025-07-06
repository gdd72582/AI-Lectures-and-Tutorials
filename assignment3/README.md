# Assignment 3

## Overview
- Student performance 데이터셋 기반 이진 분류 문제 (Pass/Fail 예측)
- 다양한 전처리 방식과 feature engineering, 여러 ML 모델 실험

---

## 🧪 Statistics

| 항목               | 내용                                     |
|------------------|----------------------------------------|
| 사용한 모델        | Decision Tree, Random Forest, SVM, LightGBM, XGBoost |
| 데이터셋           | Student performance dataset             |
| 최고 평균 성능    | 0.9922 (XGBoost, feature selection 포함) |
| 총 실험 횟수     | 45회                                   |

---

## 💡 Key Insights

### 📄 [실험 관리]
- **Google Sheet**로 실험 번호, 실험 조건, 메모를 기록하여 실험 비교와 분석이 훨씬 용이.
- 실험 결과 관리가 체계화되어, 가장 성능이 좋은 모델 및 설정을 빠르게 파악 가능.

### 📊 [학습 및 성능 분석]
- **데이터 전처리의 민감함**: 결측치 및 이상치를 제거했을 때 오히려 성능 하락을 경험 → 정보 손실과 오버피팅 가능성에 대한 고민.
- **데이터 분포 분석의 중요성**: 결측치, 이상치, 중복치의 원인과 위치를 철저히 분석하며 실험 설계의 기초로 삼음.
- **Feature Engineering**: 변수 간 correlation 분석을 통해 새로운 feature (예: Score_Gain)를 도출하고, feature selection을 통해 성능을 극대화.
- **Hyperparameter 시각화**: 파라미터 튜닝 과정을 시각화하여 파인튜닝 방향성을 더 쉽게 이해하고 개선.

---

## 🎯 Conclusion

이번 과제를 통해 **데이터 전처리와 feature engineering이 성능 향상에 미치는 실제 영향**을 구체적으로 체험했습니다.  
모델의 복잡도보다 **데이터의 질과 feature 설계가 더 큰 영향을 미친다는 점**, 그리고 실험 기록과 분석의 중요성을 확인했습니다.  
또한, 단순히 "성능이 좋다"는 애매한 표현 대신, **구체적인 수치(예: 6.6% 성능 향상)**를 제시하는 것이 설득력 있는 결과 보고에 필요하다는 점을 배웠습니다.

---

## 💬 Additional Note

> 앞으로 데이터 분석과 실험 설계에서 **실험 기록화**, **데이터 시각화 기반 설계**, **구체적 수치 기반 결론**을 더욱 강조하여, 실험의 신뢰성과 재현성을 높일 계획입니다.