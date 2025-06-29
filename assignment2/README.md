# Assignment 2

## Overview
- Various CNN architecture experiments (AlexNet, VGG, Inception, ResNet)
- Based on CIFAR-10 dataset
- Comprehensive analysis including structural performance, computational efficiency, and gradient flow

---

## Statistics

| Item | Content |
|------|---------|
| Models Used | AlexNet, VGG, Inception, ResNet |
| Dataset | CIFAR-10 |
| Experiment Repetitions | 5 per problem |
| Best Accuracy | 94.6% (ResNet-18) |

---

## Assignment 2: CNN Architecture Experiment Preview

### Q1. AlexNet Data Augmentation Effect Verification
- **3.4%p accuracy improvement** with data augmentation (baseline 82.9% → augmented 86.3%)
- Up to **86.9%** with additional augmentation

### Q2. VGG 3×3 vs 7×7 Convolution Comparison
- 3×3 Conv: Average accuracy **86.9%**, fewer parameters and faster computation
- 7×7 Conv: Accuracy **82.5%**, approximately 2.9x more parameters, 23% longer computation time

### Q3. Inception 1×1 Conv Effect Analysis
- Higher average accuracy without 1×1 Conv (**81.4% vs 79.9%**) and shorter computation time
- 2x more parameters in non-1×1 Conv model → trade-off exists

### Q4. VGG/Inception Layer Depth Comparison (16 vs 50 layers)
- VGG-50: Gradient vanishing → sharp accuracy drop (**84.3% → 38.7%**)
- Inception-50: Stable gradient flow, maintained accuracy (**around 76%**)

### Q5. ResNet Performance Comparison by Depth
- ResNet-18: 94.6%, ResNet-50: 93.5%, ResNet-101: 93.7%
- Minimal performance degradation with increased depth, gradient maintained thanks to skip connections
- Minimal performance benefits compared to increased computation and parameters → ResNet-18 most efficient

---

> This preview summarizes the key ideas and experimental results of Assignment 2. For detailed analysis and numerical values, please refer to individual README and reports.

## Key Insights

### Model Structural/Theoretical Techniques
- **Data Augmentation**: Approximately 3.4%p accuracy improvement in AlexNet experiment, up to 86.9% average with additional augmentation → data diversity contributes to generalization performance.
- **3×3 Conv vs 7×7 Conv (VGG)**: Stacking multiple small kernels (3×3) shows higher performance, fewer parameters, and faster computation speed.  
  - VGG-3×3: Average accuracy 86.9%, VGG-7×7: 82.5%
  - Approximately 2.9x difference in parameter count, about 23% faster computation speed.
- **1×1 Conv (Inception)**: 1×1 Conv for channel reduction is effective for parameter reduction but doesn't always lead to performance improvement and computation time reduction.  
  - Actually, the model without 1×1 Conv showed higher average accuracy at 81.4% and shorter computation time.
- **Layer Depth vs Performance**:
  - VGG-50 shows sharp accuracy drop due to gradient vanishing compared to VGG-16 (84.3% → 38.7%).
  - Inception-50 maintains similar accuracy to Inception-v1 and shows stable gradient flow.
- **ResNet Experiment**:
  - ResNet-18, 50, 101 all achieve high accuracy levels of 93~94%.
  - Performance doesn't sharply decline even with increased depth, and gradient flow is maintained thanks to skip connections.
  - However, ResNet-50, 101 show minimal performance improvement compared to increased computation and parameters → ResNet-18 is most efficient.

---

## Conclusion

Through this assignment, I realized that **CNN architecture design is not simply about increasing depth or reducing parameters**.  
- **Detailed structural decisions** are needed including data augmentation, kernel size selection, channel reduction design, residual structure, etc.,
- Particularly, I could specifically confirm layer-by-layer learning contributions through gradient flow visualization analysis.
- Through experiments, I directly experienced the "gap between theory and practice" and deeply understood the importance of balancing structural efficiency and accuracy.

---

## Additional Note

> In future model design, I plan to prioritize structural efficiency and data suitability rather than simply increasing complexity. I also plan to actively utilize debugging through gradient flow analysis.