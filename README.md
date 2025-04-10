
# 🐾 Multi-Class Animal Classification

This project implements a Convolutional Neural Network (CNN) to classify images of animals into multiple categories using TensorFlow and Keras.

## 🚀 Getting Started on Google Colab

Follow these steps to run the notebook efficiently using GPU acceleration:

### 1. Open in Google Colab
- Upload the notebook `Multi_Class_Animal_Classification_Week 2.ipynb` to your Google Drive.
- Open it with Google Colab.

### 2. Change Runtime to GPU (T4 Recommended)
- Go to the menu bar and select: `Runtime` > `Change runtime type`
- Set **Hardware accelerator** to `GPU` (preferably T4 for faster training)
- Click `Save`

### 3. Run the Notebook
- Execute each cell sequentially.
- The model will load the dataset, preprocess it, build the CNN, train it, and visualize performance metrics.

## 📁 Project Structure

- **Data Preprocessing**: Loads and augments the dataset using `ImageDataGenerator`
- **Model Building**: Constructs a custom CNN using Keras Sequential API
- **Training**: Compiles and fits the model on the dataset
- **Evaluation**: Evaluates performance using accuracy and loss metrics
- **Visualization**: Displays training and validation curves

## ⚡ Tips for Better Performance
- Using a GPU (like Tesla T4) significantly speeds up training compared to CPU.
- Ensure data augmentation is applied for better generalization.

---

Feel free to fork this repo, modify the model, and experiment with different architectures or datasets!
