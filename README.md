
<h1 align="center">🐾 Multi-Class Animal Recognition (MCAR)</h1>

<p align="center">
A deep learning-powered application that classifies 90 different animal species using TensorFlow, MobileNetV2, and Streamlit.
</p>

---

## 📌 Project Overview

The **Multi-Class Animal Recognition (MCAR)** system enables users to train a deep learning model using a dataset of 90 animal categories, and predict the species of an uploaded image. With a seamless UI, transfer learning model, and GPU acceleration, MCAR is a powerful tool for educational and practical use cases in computer vision.

---

## ✅ Key Features

- 🎯 **Image Classification**: Predicts from 90 animal categories.
- 🚀 **Transfer Learning**: Built on **MobileNetV2** for high accuracy and efficiency.
- 📊 **Training Visualization**: Real-time graphs of accuracy and loss.
- 🧠 **Model Training**: Streamlit interface to train your own model.
- 🔍 **Live Prediction**: Upload any animal image and get predictions.
- 💾 **Model Saving**: Automatically saves trained model and class names.
- 📥 **Dataset Downloader**: One-click KaggleHub integration.

---

## 📜 Prerequisites

- Python 3.7+

### Required Libraries

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Optional

- `scikit-learn`: For classification reports
- GPU: Supports TensorFlow GPU for faster training

---

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/mcar-animal-classifier.git
cd mcar-animal-classifier
```

### 2️⃣ (Optional) Setup a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🔧 Application Modes

### 🏋️ Train & Download Mode

- Downloads the animal dataset using KaggleHub.
- Allows configuration of **batch size** and **epochs**.
- Trains and saves the model (`MCAR.keras`) and class names.

### 🐶 Predict Mode

- Upload an image (jpg/jpeg/png).
- Get a predicted animal class with confidence score.

---

## 📂 File Structure

```bash
mcar-animal-classifier/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── MCAR.keras             # Trained model
├── class_names.txt        # Class label names
├── README.md              # Project documentation
```

---

## 🤖 Built With

- **Python 3.7+**
- **TensorFlow & Keras**
- **MobileNetV2** (pretrained model)
- **Streamlit** (for interactive UI)
- **Matplotlib & Pandas** (visualization & tracking)
- **KaggleHub** (dataset download)

---

## 🚚 Deployment

You can run this locally using Streamlit or deploy it on:

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Heroku](https://heroku.com)
- [AWS EC2 / GCP / Azure]

---

## 📸 Screenshots

| Train Mode | Predict Mode |
|------------|---------------|
| ![image](https://github.com/user-attachments/assets/b2aa026c-7a54-4830-811b-65d8d998bd7c)
|![image](https://github.com/user-attachments/assets/5c818425-7aa1-4d8e-bd16-0935368ec2d3) |


---

## 🤝 Acknowledgments

- [Sourav Banerjee](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) for the animal dataset
- TensorFlow & Streamlit community for documentation and support

---

## 📧 Contact

- GitHub: [Ronak Bansal]([https://github.com/yourusername](https://github.com/Ronak1231))
- Email: ronakbansal12345@gmail.com

---

> ⭐ If you like this project, don’t forget to **star it on GitHub**!
