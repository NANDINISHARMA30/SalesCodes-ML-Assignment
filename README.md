# 🚀 Spot the Fake Photo

### SalesCode AI – Computer Vision & Machine Learning Take-Home Assignment

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

</p>

---

## 📌 Problem Statement

Fraudsters can bypass identity verification systems by **photographing another screen** (phone, laptop, tablet, or monitor) instead of capturing the real object.

The challenge is to build a lightweight Computer Vision pipeline capable of distinguishing:

* 📷 **Real Photo**
* 💻 **Photo of a Screen (Recaptured Image)**

The final system outputs a fraud probability between **0 and 1**.

Example:

```bash
python predict.py image.jpg

0.94
```

where:

* **0 → Real Photo**
* **1 → Screen Photo**

---

# 🎯 Objective

The assignment emphasizes building a solution that is:

* Accurate
* Fast
* Lightweight
* Mobile Friendly
* Production Ready

rather than simply using the largest deep learning model.

---

# 🏗 Solution Overview

Instead of relying solely on deep neural networks, this project combines **classical Computer Vision**, **frequency-domain analysis**, and **Machine Learning** to detect subtle artifacts introduced when photographing digital displays.

The pipeline is designed to be efficient enough for deployment on mobile devices while maintaining high predictive performance.

---

# 📂 Dataset

Following the assignment guidelines, a custom dataset was collected containing approximately:

| Class         | Images |
| ------------- | ------ |
| Real Photos   | ~50    |
| Screen Photos | ~50    |

The dataset includes variations in:

* Indoor & outdoor lighting
* Different screen brightness levels
* Multiple viewing angles
* Phones, laptops, tablets, and monitors
* Various object categories and backgrounds

This diversity helps improve generalization to unseen images.

---

# ⚙ Feature Engineering

Each image is transformed into a handcrafted feature vector capturing characteristics that commonly distinguish recaptured images.

The extracted features include:

* Image sharpness
* Edge density
* Brightness statistics
* Contrast
* Texture descriptors
* Color statistics
* Frequency-domain (FFT) features
* Screen artifact indicators
* Reflection and glare characteristics

A total of **42 handcrafted features** are extracted from every image.

---

# 🤖 Models Evaluated

Rather than relying on a single algorithm, multiple machine learning models were trained and benchmarked.

## 📊 Model Benchmark Results

The training pipeline benchmarks multiple classical machine learning algorithms and lightweight deep feature extraction approaches. The best-performing model is automatically selected based on **accuracy, ROC-AUC, model size, and inference latency**.

| Model                                |  Accuracy |  Precision |   Recall  |  F1 Score |  ROC-AUC  | Inference (ms) | Model Size (MB) |
| :----------------------------------- | :-------: | :--------: | :-------: | :-------: | :-------: | :------------: | :-------------: |
| **🏆 Logistic Regression**           | **96.3%** | **100.0%** | **92.9%** | **96.3%** | **96.2%** |    **0.07**    |    **0.007**    |
| Extra Trees                          |   92.6%   |   100.0%   |   85.7%   |   92.3%   |   96.7%   |      6.29      |       3.25      |
| Gradient Boosting                    |   92.6%   |   100.0%   |   85.7%   |   92.3%   |   96.2%   |      0.08      |       1.13      |
| HistGradientBoosting                 |   92.6%   |   100.0%   |   85.7%   |   92.3%   |   94.0%   |      0.56      |       0.42      |
| SVM (RBF)                            |   92.6%   |   100.0%   |   85.7%   |   92.3%   |   96.7%   |      0.21      |       0.05      |
| XGBoost                              |   92.6%   |   100.0%   |   85.7%   |   92.3%   |   96.2%   |      0.23      |       0.94      |
| Random Forest                        |   88.9%   |    92.3%   |   85.7%   |   88.9%   |   95.6%   |      6.25      |       1.24      |
| AdaBoost                             |   88.9%   |   100.0%   |   78.6%   |   88.0%   |   96.2%   |      3.01      |       0.49      |
| CatBoost                             |   88.9%   |   100.0%   |   78.6%   |   88.0%   |   95.6%   |      0.14      |       1.36      |
| MobileNetV3 Embeddings + XGBoost     |   88.9%   |   100.0%   |   78.6%   |   88.0%   |   91.2%   |      89.99     |       4.36      |
| K-Nearest Neighbors                  |   85.2%   |   100.0%   |   71.4%   |   83.3%   |   94.5%   |      1.36      |       0.03      |
| LightGBM                             |   85.2%   |   100.0%   |   71.4%   |   83.3%   |   94.5%   |      0.27      |       0.46      |
| MLP Classifier                       |   81.5%   |    90.9%   |   71.4%   |   80.0%   |   91.2%   |      0.07      |       0.19      |
| EfficientNet-B0 Embeddings + XGBoost |   85.2%   |   100.0%   |   71.4%   |   83.3%   |   94.5%   |     193.05     |      16.69      |

### 🏅 Final Model Selection

After evaluating all models, **Logistic Regression** was selected as the final production model because it achieved the highest validation accuracy while also providing:

*  **Highest Accuracy:** **96.3%**
*  **Fastest Inference:** **0.07 ms/image**
*  **Tiny Model Size:** **0.007 MB**
*  **Mobile-Friendly Deployment**
*  **Near-zero inference cost (on-device)**

Although pretrained feature extractors such as **MobileNetV3** and **EfficientNet-B0** were evaluated, they introduced significantly higher latency and larger model sizes without improving validation accuracy on this dataset. Therefore, the lightweight Logistic Regression model was selected as the most practical solution for deployment.


---


# 📊 Performance

| Metric         |             Score |
| -------------- | ----------------: |
| Accuracy       |         **96.3%** |
| Precision      |          **100%** |
| Recall         |         **92.9%** |
| F1 Score       |         **96.3%** |
| ROC-AUC        |         **0.962** |
| Inference Time | **0.07 ms/image** |
| Model Size     |      **0.007 MB** |

> **Note:** Results are reported on a held-out validation split generated from the collected dataset. The assignment will evaluate the final `predict.py` on unseen images.

---

# ⚡ Performance Comparison

The project automatically benchmarks all supported models and selects the best candidate based on:

* Validation Accuracy
* ROC-AUC
* Model Size
* Inference Latency

This ensures the deployed model is both accurate and practical.

---

# 📁 Project Structure

```text
SalesCodes-ML-Assignment/

│── dataset/
│   ├── real/
│   └── screen/
│
│── features.py
│── train.py
│── predict.py
│── model.pkl
│── results/
│── requirements.txt
│── README.md
```

---

# 🚀 Installation

```bash
git clone https://github.com/NANDINISHARMA30/SalesCodes-ML-Assignment.git

cd SalesCodes-ML-Assignment

pip install -r requirements.txt
```

---

# 🏃 Training

```bash
python train.py
```

The training pipeline automatically:

* Loads the dataset
* Extracts handcrafted features
* Trains multiple models
* Benchmarks performance
* Selects the best model
* Calibrates probabilities
* Saves the final model as `model.pkl`

---

# 🔍 Prediction

```bash
python predict.py image.jpg
```

Example output:

```text
0.91
```

Interpretation:

| Output | Meaning      |
| ------ | ------------ |
| 0      | Real Photo   |
| 1      | Screen Photo |

---

# 📱 Deployment

This solution is designed for deployment on edge devices.

| Property        | Value                                  |
| --------------- | -------------------------------------- |
| Runs Offline    | ✅                                      |
| CPU Friendly    | ✅                                      |
| Mobile Friendly | ✅                                      |
| Cloud Required  | ❌                                      |
| Model Size      | ~0.007 MB                              |
| Average Latency | ~0.07 ms/image                         |
| Cost per Image  | Approximately $0 (on-device inference) |

---

# 🔬 Future Improvements

With additional time and data, the following improvements could further enhance robustness:

* Larger and more diverse dataset
* K-fold cross-validation
* Hybrid handcrafted + CNN feature fusion
* ONNX/TFLite export for mobile deployment
* Adaptive fraud threshold selection
* Continuous retraining using newly collected fraud samples
* Explainability using SHAP or feature importance analysis

---

# 🛠 Tech Stack

* Python
* OpenCV
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Joblib

---


## 👩‍💻 Author

**Nandini Sharma**


