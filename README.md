# README for Stress Detection Research

## Project Title: **Machine Learning Algorithms for Detecting Mental Stress in College Students**

### Authors:
- Ashutosh Singh ([ashutoshs22102@iiitnr.edu.in](mailto:ashutoshs22102@iiitnr.edu.in))
- Khushdeep Singh ([khushdeep22102@iiitnr.edu.in](mailto:khushdeep22102@iiitnr.edu.in))
- Amit Kumar ([amit22102@iiitnr.edu.in](mailto:amit22102@iiitnr.edu.in))
- Abhishek Shrivastava ([abhisheks@iiitnr.edu.in](mailto:abhisheks@iiitnr.edu.in))
- Santosh Kumar ([santosh@iiitnr.edu.in](mailto:santosh@iiitnr.edu.in))

### Institution:
Department of Data Science and Artificial Intelligence, IIIT Naya Raipur, Chhattisgarh, India.

---

## Abstract:
This project focuses on predicting and mitigating mental stress among college students using machine learning algorithms. The study leverages a dataset collected through a stress survey validated by medical experts, with 843 participants aged 18-21 years. Seven machine learning algorithms were employed, with Support Vector Machines (SVM) achieving the highest accuracy of 95%.

---

## Repository Structure:
```
root/
|-- data/               # Contains dataset files in CSV format.
|-- src/                # Source code for preprocessing, modeling, and evaluation.
|-- results/            # Results, including plots, confusion matrices, and performance metrics.
|-- models/             # Saved machine learning models.
|-- README.md           # Project documentation (this file).
```

---

## Dataset:
The dataset was collected via a Google Form and contains responses from 843 college students. It includes 28 features categorized into:
1. Emotional Well-being
2. Physical Health
3. Academic Performance
4. Relationships
5. Leisure Activities
6. Stress and Non-stress levels

**Key Dataset Information:**
- **Size:** 843 entries
- **Attributes:** 28 features + labels (Stress/Non-stress)
- **Format:** CSV

Dataset Link: [Stress and Well-being Data of College Students](https://www.kaggle.com/datasets/ashutoshsingh22102/stress-and-well-being-data-of-college-students)

---

## Methodology:
The proposed framework involves:
1. **Data Collection**: Survey-based real-time data collection.
2. **Data Preprocessing**: Duplicate handling, encoding, normalization.
3. **Feature Extraction**: Selection of stress-relevant attributes.
4. **Model Training and Testing**: 75% training, 25% testing split.
5. **Algorithms Used**:
   - Decision Trees
   - Random Forest
   - Support Vector Machines
   - AdaBoost
   - Naive Bayes
   - Logistic Regression
   - K-Nearest Neighbors
6. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC Curve.

---

## Results:
**Algorithm Performance Summary:**
| Algorithm           | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Decision Trees      | 87%      | 0.85      | 0.86   | 0.85     |
| Random Forest       | 92%      | 0.91      | 0.90   | 0.90     |
| **Support Vector Machines** | **95%**  | **0.94**  | **0.93** | **0.94**  |
| AdaBoost            | 89%      | 0.88      | 0.89   | 0.88     |
| Naive Bayes         | 86%      | 0.84      | 0.85   | 0.84     |
| Logistic Regression | 88%      | 0.87      | 0.88   | 0.87     |
| K-Nearest Neighbors | 85%      | 0.83      | 0.84   | 0.83     |

**Key Findings:**
- Support Vector Machines (SVM) achieved the highest accuracy (95%).
- The confusion matrix and ROC curve illustrate effective classification of stress vs. non-stress.

---

## Installation and Usage:
### Prerequisites:
- Python 3.8+
- Required libraries:
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  ```

### Steps to Run:
1. Clone the repository:
   ```bash
   git clone https://github.com/username/stress-detection.git
   cd stress-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing script:
   ```bash
   python src/preprocess.py
   ```
4. Train and evaluate models:
   ```bash
   python src/train_models.py
   ```
5. View results and visualizations in the `results/` folder.

---

## Citation:
If you use this work, please cite:
```
@article{StressDetection2024,
  title={Machine Learning Algorithms for Detecting Mental Stress in College Students},
  author={Ashutosh Singh, Khushdeep Singh, Amit Kumar, Abhishek Shrivastava, and Santosh Kumar},
  journal={arXiv preprint arXiv:2412.07415},
  year={2024}
}
```

---

## Acknowledgments:
- **Medical Guidance**: Experts from AIIMS Raipur.
- **Dataset Participants**: College students aged 18-21 years.
- **Funding and Support**: IIIT Naya Raipur.

---

## License:
This project is licensed under the MIT License. See the LICENSE file for details.
