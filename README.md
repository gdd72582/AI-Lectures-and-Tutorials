# AI Lecture and Tutorials

> Artificial Intelligence and Machine Learning Course (2025)

This repository contains assignments and exercises completed during a 2025 undergraduate course on Artificial Intelligence and Machine Learning.

Each task focuses on building a solid understanding of core ML concepts through both theory and implementation.  
Using libraries like PyTorch and Scikit-learn, topics range from basic classification to deep learning with CNNs.

Assignments are organized by topic and model type, and include:
- Data preprocessing and exploratory analysis  
- Model training and evaluation  
- Neural network architectures (MLP, CNN)  
- Comparative experiments and performance analysis


## Assignments

### Assignment 1

Foundations of AI and Deep Learning

| No. | Topic |
|:---:|:------|
| Q1 | Relationship among AI, Machine Learning, and Deep Learning + Rise of Deep Learning since 2012 |
| Q2 | Overview of Deep Learning Training Pipeline |
| Q3 | Explanation of Gradient, Backpropagation, and Chain Rule |
| Q4 | Comparison of Activation Functions: ReLU vs Sigmoid |
| Q5 | Comparison of Batch Normalization vs Group Normalization |
| Q6 | Performance Comparison Based on the Order of BatchNorm and Activation |

---

### Assignment 2

CNN Architecture Experiments

| No. | Topic |
|:---:|:------|
| Q1 | **AlexNet**: Impact of Data Augmentation on Classification Performance |
| Q2 | **VGG**: Comparison of 3×3 vs 7×7 Convolutions (Params, Runtime, Accuracy) |
| Q3 | **Inception**: Analysis with and without 1×1 Convolution (Efficiency, Accuracy) |
| Q4 | **VGG vs Inception (50 layers)**: Deeper Network Gradient Flow and Performance Comparison |
| Q5 | **ResNet**: Performance Comparison of ResNet-18, ResNet-50, and ResNet-101 |

---

### Assignment 3

ML Assignment 1: Binary Classification

| No. | Topic |
|:---:|:------|
| Q1 | Data Preprocessing and EDA: Variable Types, Missing Values, Outlier Handling, Visual Analysis |
| Q2 | Model Training and Evaluation: Decision Tree, Random Forest, and SVM (with 5-Fold CV) |
| Q3 | Hyperparameter Tuning: GridSearch / RandomSearch with 2+ Parameters |
| Q4 | Final Test Accuracy Evaluation with Best Model |
| Q5 | Extra Credit: XGBoost Implementation and Performance Evaluation (≥87% Accuracy)

---

### Assignment 4

ML Assignment 2: Image Classification using MLP

| No. | Topic |
|:---:|:------|
| Q1 | Visualization of Representative Images for All Classes (CIFAR-10 & FashionMNIST) |
| Q2 | Dataset Complexity Comparison Based on MLP Architecture |
| Q3 | Model Design & Performance on CIFAR-10: Hidden Layers, Dropout, Activation Tests |
| Q4 | Model Design & Performance on FashionMNIST: Architecture Reuse & Evaluation |
| Q5 | Strategy Summary: Best Performing Configurations and Reasoning

---

### Assignment 5

ML Assignment 3: Binary Classification with CNN (Smoke Image Dataset)

| No. | Topic |
|:---:|:------|
| Q1 | Smoke Dataset Preparation: Path Mapping, Labeling, DataLoader Implementation |
| Q2 | CNN Model Design & Training: Use of `BCELoss` with Custom Architecture |
| Q3 | Early Stopping & Learning Curve Visualization (Loss, Accuracy) |
| Q4 | Evaluation: Accuracy and F1-Score on Test Set, Model Saving |
| Q5 | Architecture Optimization: Effects of Layers, Dropout, Optimizers, Learning Rate

---

## Learning Statistics

| Item | Count |
|:----:|:----:|
| **Total Assignments** | 5 |
| **Total Problems** | 25 |
| **Total Experiments** | 306 |
| **Average Experiment Repetitions** | 5 times |
| **Total Learning Time** | 160+ hours |
| **Models Experimented** | 20+ (CNN, MLP, Tree-based, SVM, etc.) |
| **Datasets Used** | 6 (CIFAR-10, FashionMNIST, STL-10, Smoke, Student Performance, etc.) |

---

## Key Insights

### Coding and Experiment Design
- **Importance of Softcoding**: Managing hyperparameters as variables for easy viewing and modification
- **Importance of Pre-experiment Data Visualization**: Actively utilizing visualization (T-SNE, scatter plots, etc.) to understand data complexity

### Experiment Management and Documentation
- **Google Sheets Utilization**: Systematic recording of experiment numbers, conditions, and notes for comparative analysis and reproducibility
- **Ensuring Reliability**: Conducting 5+ experiments with different seeds and using average performance as final results
- **Specific Numerical Expression**: Using precise numbers like "6.6% performance improvement" instead of vague terms like "good"

### Performance Analysis and Visualization
- **Gradient Vanishing Analysis**: Confirming differences in gradient flow and performance through ReLU vs Sigmoid comparison experiments
- **Data Complexity Assessment**: Evaluating dataset characteristics and difficulty through visualization (t-SNE, scatter plots, etc.)
- **Misclassification Sample Analysis**: Understanding model limitations through actual error pattern analysis rather than simple accuracy
- **Learning Curve Interpretation**: Understanding the importance of distinguishing underfitting from overfitting through failed interpretation experiences

### Model Architecture and Optimization
- **Data Augmentation Effects**: Confirming actual effects (e.g., 3.4%p improvement in AlexNet)
- **BatchNorm vs GroupNorm**: Understanding performance differences and batch size sensitivity
- **Residual Learning**: Solving gradient vanishing problems in deep networks through Residual Learning and Bottleneck Block design
- **Transfer Learning**: 23%p performance improvement when using Pretrained ResNet-50 (72% → 95%)
- **1×1 Convolution**: Experimental verification of parameter reduction effects and performance trade-offs

### Data Preprocessing and Feature Engineering
- **Preprocessing Sensitivity**: Performance changes due to outlier and missing value handling → balancing information loss and overfitting
- **Feature Engineering**: Creating new features through correlation analysis between variables (e.g., Score_Gain)
- **Data Distribution Analysis**: Confirming the impact of data distribution analysis and visualization-based design on experiment performance

### Personal Growth Experience
- **Theory vs Practice Gap**: Prioritizing structural efficiency and data suitability over simply increasing complexity
- **Objectivity in Experiment Design**: Understanding the importance of diverse model approaches through underestimating Logistic Regression
- **Limitation Recognition**: Feeling limitations with model modifications alone in STL-10, leading to introduction of pseudo-labeling and ensemble techniques
- **Learning from Failures**: Understanding the importance of monitoring learning processes through failed loss curve interpretation
- **Time Management Importance**: Recognizing the need for efficient experiment design and priority setting through 160+ hours of experiments
- **Documentation Habits**: Recognizing the importance of systematically recording experiment processes and results

---

## Conclusion

Through this AI/ML course, I deeply experienced the importance of **balancing theory and practice** and **systematic experiment design**.

### Core Learning Outcomes
- Confirmed that **systematic repeated experiments and documentation are key to performance improvement** through 306 experiments over 160+ hours
- Experienced that **feature engineering has greater impact than model complexity** through direct comparison of various model architectures
- Directly experimented with **Transfer Learning's actual performance improvement effects** and the importance of structural design
- Recognized the necessity of **specific numerical-based result expression** and visualization for reproducible research

### Future Plans
- Ensuring research reliability by combining **objective result analysis**, **sufficient repeated experiments**, and **visualization-based insights**
- Applying various transfer learning strategies including pretrained backbone and custom head combinations
- Model debugging and structural optimization through gradient flow analysis
- Strengthening experiment reliability and reproducibility through specific numerical-based conclusions

I will utilize this experience in more complex deep learning models and real-world projects to develop practical capabilities that achieve both performance and efficiency simultaneously.

---
