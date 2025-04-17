import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import kagglehub
import streamlit as st
from PIL import Image

# Attempt to import classification_report
try:
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed. Classification report will be skipped. To enable, install scikit-learn via `pip install scikit-learn`.")

# Initialize session state keys
state_keys = ['dataset_path', 'data_dir', 'class_names', 'history_df']
for key in state_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Helper functions ---
def download_dataset(dataset_ref: str) -> str:
    path = kagglehub.dataset_download(dataset_ref)
    return path


def configure_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        return True
    return False


def build_data_generators(dataset_path: str, img_size: tuple, batch_size: int):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_gen, val_gen


def build_model(num_classes: int, input_shape: tuple):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(data_dir, img_size, batch_size, epochs):
    train_gen, val_gen = build_data_generators(data_dir, img_size, batch_size)
    class_names = list(train_gen.class_indices.keys())
    model = build_model(len(class_names), input_shape=(img_size[0], img_size[1], 3))
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        epochs=epochs
    )
    model.save('MCAR.keras')
    # save class names
    with open('class_names.txt', 'w') as f:
        f.write("\n".join(class_names))

    # Convert history to DataFrame and store in session state
    hist_df = pd.DataFrame({
        'epoch': list(range(1, len(history.history['accuracy']) + 1)),
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
    })
    st.session_state.history_df = hist_df
    return history, class_names


def predict_image(model_path: str, class_names: list, img) -> tuple:
    model = load_model(model_path)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    return class_names[idx], preds[0][idx]

# --- Streamlit App ---
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🐾 Multi-Class Animal Recognition")

mode = st.sidebar.selectbox("Choose Mode", ['Train & Download', 'Predict'])

if mode == 'Train & Download':
    st.header("Download Dataset & Train Model")
    if st.button("🔄 Download Dataset"):
        st.session_state['dataset_path'] = download_dataset("iamsouravbanerjee/animal-image-dataset-90-different-animals")
        st.session_state['data_dir'] = os.path.join(st.session_state['dataset_path'], 'animals/animals')
        st.success(f"Dataset downloaded to: {st.session_state['dataset_path']}")

    enable_gpu = st.checkbox("Enable GPU memory growth")
    if enable_gpu:
        if configure_gpu_growth():
            st.success("GPU configured for memory growth.")
        else:
            st.warning("No GPU detected; running on CPU.")

    epochs = st.slider("Epochs", min_value=1, max_value=50, value=20)
    img_size = (224, 224)
    batch_size = st.number_input("Batch size", min_value=8, max_value=128, value=64)

    if st.button("🚀 Train Model"):
        if not st.session_state['data_dir']:
            st.error("Please download the dataset first.")
        else:
            with st.spinner("Training in progress..."):
                history, class_names = train_model(
                    st.session_state['data_dir'], img_size, batch_size, epochs
                )
                st.session_state['class_names'] = class_names
            st.success("Model trained and saved as MCAR.keras")
            # Plot history
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['accuracy'], label='Train Acc')
            ax[0].plot(history.history['val_accuracy'], label='Val Acc')
            ax[0].set_title('Accuracy')
            ax[1].plot(history.history['loss'], label='Train Loss')
            ax[1].plot(history.history['val_loss'], label='Val Loss')
            ax[1].set_title('Loss')
            for a in ax:
                a.legend()
            st.pyplot(fig)

            # Display accuracy table
            if st.session_state.history_df is not None:
                st.subheader("Training History")
                st.table(st.session_state.history_df)

else:
    st.header("Predict Animal from Image")
    uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption='Uploaded Image', use_container_width=True)
        if os.path.exists('MCAR.keras') and os.path.exists('class_names.txt'):
            with open('class_names.txt', 'r') as f:
                class_names = f.read().splitlines()
            pred_class, confidence = predict_image('MCAR.keras', class_names, pil_img)
            st.success(f"Predicted: **{pred_class}** with confidence {confidence:.2f}")
        else:
            st.error("Model or class_names.txt not found. Please train the model first.")
