# 🩺 Skin Cancer Classification using Deep Learning (CNN)

## 📌 Project Overview

Skin cancer is one of the most common forms of cancer worldwide. Early detection significantly increases the chances of successful treatment. This project presents a **Convolutional Neural Network (CNN)** model developed using **TensorFlow** and **Keras** to automatically classify skin lesion images into two categories:

* **Benign (Non-Cancerous)**
* **Malignant (Cancerous)**

The model is trained on a dataset of dermoscopic skin images using a deep learning approach. Image preprocessing, CNN architecture, model checkpointing, and performance visualization are implemented to achieve reliable classification accuracy.

---

## 🎯 Objectives

* Build an automated skin cancer classification system.
* Perform image preprocessing and normalization.
* Train a Convolutional Neural Network (CNN).
* Save the best-performing model during training.
* Evaluate model performance using accuracy and loss curves.

---

# 🛠️ Technologies Used

* Python 3.12
* TensorFlow
* Keras
* NumPy
* Matplotlib
* ImageDataGenerator

---

# 📂 Dataset Structure

The dataset should be organized as follows:

```
skin_dataset_resized/
│
├── train_set/
│   ├── Benign/
│   └── Malignant/
│
└── val_set/
    ├── Benign/
    └── Malignant/
```

Update the dataset paths inside the code before running:

```python
train_dir = r'path_to_train_set'
validation_dir = r'path_to_validation_set'
```

---

# 📷 Image Preprocessing

The dataset is loaded using **ImageDataGenerator**.

The preprocessing includes:

* Rescaling pixel values from **0–255** to **0–1**
* Image resizing to **150 × 150 pixels**
* Batch size of **32**
* Binary classification mode

```python
ImageDataGenerator(rescale=1./255)
```

---

# 🧠 CNN Architecture

The proposed CNN consists of:

| Layer      | Details                             |
| ---------- | ----------------------------------- |
| Input      | 150 × 150 × 3 RGB Image             |
| Conv2D     | 32 Filters (3×3), ReLU              |
| MaxPooling | 2×2                                 |
| Conv2D     | 64 Filters (3×3), ReLU              |
| MaxPooling | 2×2                                 |
| Conv2D     | 128 Filters (3×3), ReLU             |
| MaxPooling | 2×2                                 |
| Conv2D     | 256 Filters (3×3), ReLU             |
| MaxPooling | 2×2                                 |
| Conv2D     | 512 Filters (3×3), ReLU             |
| MaxPooling | 2×2                                 |
| Flatten    | Converts feature maps into a vector |
| Dense      | 128 neurons, ReLU                   |
| Dropout    | 50%                                 |
| Output     | 1 neuron, Sigmoid                   |

---

# ⚙️ Model Compilation

The model uses:

* **Optimizer:** Adam
* **Learning Rate:** 0.0001
* **Loss Function:** Binary Crossentropy
* **Evaluation Metric:** Accuracy

```python
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

# 💾 Model Checkpoint

The best model is automatically saved whenever the validation loss improves.

```python
ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
```

---

# 🚀 Model Training

Training configuration:

* Epochs: **15**
* Batch Size: **32**
* Optimizer: **Adam**
* Validation Data Included
* Best Model Saved Automatically

```python
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)
```

---

# 📊 Training Results

Training Summary

* Training Accuracy reached approximately **89%**
* Validation Accuracy reached approximately **88%**
* Lowest Validation Loss: **0.2033**
* Best model saved as **best_model.keras**

These results indicate that the CNN successfully learned meaningful image features for distinguishing between benign and malignant skin lesions.

---

# 📈 Performance Visualization

The project visualizes:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

using **Matplotlib**.

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```

and

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```

These graphs help monitor learning progress and identify potential overfitting or underfitting.

---

# 📌 Output

The trained model predicts whether a skin lesion belongs to:

* **Benign**
* **Malignant**

The final trained model is stored as:

```
best_model.keras
```

which can later be loaded for prediction on unseen skin lesion images.

---

# 📋 Future Improvements

The current implementation can be enhanced by:

* Applying data augmentation (rotation, flipping, zooming, brightness adjustment)
* Using transfer learning models such as EfficientNet, ResNet50, MobileNetV2, or DenseNet121
* Increasing the dataset size
* Hyperparameter tuning
* Implementing early stopping
* Adding precision, recall, F1-score, confusion matrix, and ROC-AUC evaluation metrics
* Building a Flask or Streamlit web application for real-time prediction

---

# 📌 Conclusion

This project demonstrates the application of Convolutional Neural Networks (CNNs) for binary skin cancer classification. The model effectively learns discriminative features from dermoscopic images and achieves approximately **89% training accuracy** with **88% validation accuracy**. By combining image preprocessing, deep CNN architecture, model checkpointing, and performance visualization, the system provides a reliable foundation for automated skin cancer detection. Future enhancements through transfer learning, data augmentation, and deployment as a web application can further improve accuracy and real-world usability.
