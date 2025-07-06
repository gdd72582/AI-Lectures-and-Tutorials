# Assignment 5

## Overview
- Smoke Image Dataset을 활용한 CNN 기반 이진 분류 실험
- 다양한 구조(PlainCNN, ResNet-50, Pretrained ResNet-50) 및 bottleneck block 설계
- Pretrained 모델과 직접 설계 모델 간 성능 비교

---

## 🧪 Statistics

| 항목             | 내용                                   |
|----------------|------------------------------------|
| 사용한 모델      | PlainCNN, ResNet-50, Pretrained ResNet-50 |
| 데이터셋         | Smoke Image Dataset              |
| 최고 성능       | 95.09% (Pretrained ResNet-50, frozen) |
| 총 실험 횟수    | 30회 이상 (구조별 반복 포함)          |

---

## 💡 Key Insights

### ⚙️ [모델 구조 및 기법]
- **Residual Learning 개념**: 깊은 네트워크에서 gradient 소실 문제를 해결하기 위해 skip connection(지름길)을 도입.
- **Bottleneck Block**: 연산량을 줄이고 표현력을 유지하기 위해 1×1 → 3×3 → 1×1 conv 구조 사용, 채널을 줄이며 병목 효과 발생 → 계산 효율성과 성능 향상에 기여.
- **Pretrained 모델 효과**:
  - 직접 구현한 ResNet-50 모델: 약 72% 정확도
  - Pretrained ResNet-50(frozen): 95.09% 정확도
  - Pretrained ResNet-50(BN+FC fine-tune): 94.6% 정확도
  - 사전 학습된 feature extractor의 강력함을 체감, transfer learning의 실질적인 성능 개선 효과 확인.

---

## 🎯 Conclusion

이번 과제를 통해, **딥러닝 모델 아키텍처 설계와 transfer learning의 차이**를 실험적으로 검증했습니다.  
- Residual 구조와 bottleneck block의 중요성을 직접 체험하며, gradient 흐름 안정화의 원리를 더 깊게 이해.
- Pretrained 모델을 사용할 때, 학습 시간과 성능 측면에서 매우 큰 이점을 확인.  
- 모델의 구조 설계와 fine-tuning 전략을 조합하면 성능과 효율성을 동시에 잡을 수 있다는 점을 실험적으로 증명했습니다.

---

## 💬 Additional Note

> 향후 실험과 연구에서는 pretrained backbone과 custom head 조합, batchnorm fine-tuning 전략 등 다양한 transfer learning 기법을 활용해, 더 복잡한 이미지 분류 및 detection 과제에 도전할 계획입니다.