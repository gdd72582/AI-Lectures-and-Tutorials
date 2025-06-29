# Assignment 1

## Overview
- Experimental comparison of differences in Activation Function and Normalization structures
- Various experiments including ReLU vs Sigmoid, BatchNorm vs GroupNorm, and Activation and BN order
- Based on Custom CNN model

---

## Statistics

| Item | Content |
|------|---------|
| Total Experiments | 140 |
| Experiment Repetitions | 5 per setting |
| Total Experiment Time | ~40 hours |
| Best Accuracy | 80%+ |
| Model Used | Custom CNN |

---

## Deep Learning Core Concepts Preview

### Q1. AI, ML, DL Relationship and 2012 Background
- Organizing the hierarchical relationship: Artificial Intelligence > Machine Learning > Deep Learning
- 2012 AlexNet and ImageNet competition success: large-scale datasets, GPU development, deep neural network technology

### Q2. Deep Learning Training Process
- Iterative process of Forward pass, Loss calculation, Backpropagation, Gradient calculation and Weight updates

### Q3. Backpropagation & Chain Rule
- Meaning and calculation method of Gradient
- Sequential gradient calculation through Chain rule
- Layer-by-layer gradient calculation through multiplication of local gradient and downstream gradient

### Q4. Activation Function: ReLU vs Sigmoid
- Experimental results: In deep networks, ReLU shows performance improvement while Sigmoid shows performance degradation due to gradient vanishing
- Comparison of gradient reduction rates according to number of blocks

### Q5. Normalization: BatchNorm vs GroupNorm
- BatchNorm: High dependency on batch size, sharp performance drop with small batches
- GroupNorm: Consistent performance regardless of batch size
- Comparative analysis of standard deviation and accuracy

### Q6. Activation and BatchNorm Order Experiment
- Conv → BN → ReLU (CASE 1), Conv → ReLU → BN (CASE 2), Conv → LeakyReLU → BN (CASE 3)
- In CNN, CASE 3 (LeakyReLU → BN) achieved highest performance and lowest standard deviation
- MLP shows minimal differences across structures, CNN shows significant performance differences by order → Necessity of experiments for performance improvement in design

---

> This preview is a concise summary of key ideas and results for each problem. For detailed analysis, numerical values, and graphs, please refer to the detailed README and notebook files for each assignment.

---

## Key Insights

### Activation Function
- Confirmed that ReLU shows **improvement** in performance as block depth increases, while Sigmoid shows **decline**.
- When experimentally outputting gradients, unexpectedly found that **both ReLU and Sigmoid show reduced gradient values**.
- Despite this, ReLU maintained good gradient flow and showed performance improvement, while Sigmoid showed sharp performance degradation due to gradient vanishing.

### Experiment Design and Analysis
- Ensured result reliability by conducting **minimum 5 repetitions** and using average performance.
- Realized that using specific numerical values like **"6.6% higher performance"** instead of vague expressions like "good" or "excellent" in report writing is much clearer.

### Normalization
- Batch Normalization responds very sensitively to **batch size**, with performance sharply declining as batch size decreases.
- Group Normalization maintains consistent performance regardless of batch size, with lower standard deviation making it more stable.

### Activation & BN Order
- In experiments changing Activation and BN order, experienced the importance of **comprehensive evaluation including model type, layer depth, and learning stability** rather than just checking performance.
- In CNN, ReLU followed by BN and LeakyReLU followed by BN order showed highest performance and lowest variance.
- MLP structure showed almost no performance differences according to order.

---

## Conclusion

Through this assignment, I deeply learned about **the difference between theory and practice**, and the importance of **experiment design and analysis beyond simple performance metrics**.  
Experiments with activation functions and normalization order made me feel that there is no "fixed answer", and confirmed that results can vary depending on model structure and experimental conditions.  
Additionally, I once again felt that **statistical metrics and accurate documentation** are essential for reproducibility and reliability in experiment design.

---

## Additional Note

> I plan to make it a habit to always check **gradient flow**, conduct **detailed numerical-based evaluation**, and perform **sufficient repeated experiments** in future experiments.