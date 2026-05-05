# 🚦 Traffic Sign Recognition using CNN

This project implements a traffic sign classification system using deep learning. A Convolutional Neural Network (CNN) is built using TensorFlow to classify images of traffic signs into multiple categories. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and demonstrates practical computer vision techniques used in real-world applications such as autonomous driving and traffic monitoring.

---

## 📊 Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which contains over 50,000 images across 43 traffic sign classes.

👉 Download the dataset here:  
🔗 https://benchmark.ini.rub.de/gtsrb_news.html

---

## ⚙️ Features

- Image classification using CNN  
- Efficient batch-based data loading  
- Image preprocessing (resizing and normalization)  
- Overfitting prevention using dropout and early stopping  
- Scalable training for large datasets  

---

## 🧠 Model Architecture

- Convolutional layers for feature extraction  
- MaxPooling layers for dimensionality reduction  
- Fully connected (Dense) layers for classification  
- Softmax output layer for multi-class prediction  

---

## ▶️ How to Run

1. Download and extract the dataset  
2. Navigate to the project folder  
3. Run the script:

```bash
python traffic_signs.py train model.h5
