import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Dropout
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, EarlyStopping
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_lstm_mvp():
    logging.info("====================================")
    logging.info("  ACTION RECOGNITION: LSTM MVP")
    logging.info("====================================")
    
    # ---------------------------------------------------------
    # PASO 1: Carga y Preparación del Dataset
    # ---------------------------------------------------------
    tensor_path = os.path.join(config.OUTPUT_DIR, "LSC_dataset_unificado.npy")
    labels_path = os.path.join(config.OUTPUT_DIR, "LSC_labels.npy")
    
    if not os.path.exists(tensor_path) or not os.path.exists(labels_path):
        logging.error("Tensors not found! Ensure the Data Pipeline step finished successfully.")
        return
        
    logging.info(f"Loading tensors from {tensor_path}")
    X = np.load(tensor_path)
    y_raw = np.load(labels_path)
    
    logging.info(f"Loaded X shape: {X.shape}")
    logging.info(f"Loaded y shape: {y_raw.shape}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    logging.info(f"Unique classes found ({num_classes}): {label_encoder.classes_}")
    
    # Train/Test Split (10% test to validate flow since it's 21 samples initially)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.10, random_state=42)
    
    logging.info(f"Training set: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # ---------------------------------------------------------
    # PASO 2: Arquitectura del Modelo Secuencial (LSTM)
    # ---------------------------------------------------------
    logging.info("Building Sequential LSTM Architecture...")
    model = Sequential([
        # Input layer expects exactly (60, 1629) shape
        Input(shape=(60, 1629)),
        
        # LSTM Capa 1
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        
        # LSTM Capa 2
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        
        # LSTM Capa 3
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        
        # Capa Densa (Output)
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.summary()
    
    # ---------------------------------------------------------
    # PASO 3: Compilación y Entrenamiento
    # ---------------------------------------------------------
    logging.info("Compiling the model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Simple TensorBoard callback
    log_dir = os.path.join(config.OUTPUT_DIR, "logs")
    tb_callback = TensorBoard(log_dir=log_dir)
    
    logging.info("Starting Training (Overfitting Test - 100 Epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[tb_callback],
        verbose=1
    )
    
    # Guardar métricas de entrenamiento a imagen
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    metrics_path = os.path.join(config.OUTPUT_DIR, "training_metrics.png")
    plt.savefig(metrics_path)
    logging.info(f"Training metrics plot saved to {metrics_path}")
    
    # ---------------------------------------------------------
    # PASO 4: Evaluación y Exportación
    # ---------------------------------------------------------
    logging.info("Evaluating on Test Set...")
    predictions = model.predict(X_test)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    logging.info("Confusion Matrix:")
    print(cm)
    
    # Ver si hay confusiones reales o todo se predijo bien
    labels_true = label_encoder.inverse_transform(y_true_classes)
    labels_pred = label_encoder.inverse_transform(y_pred_classes)
    for i in range(len(y_test)):
        status = "✅" if labels_true[i] == labels_pred[i] else "❌"
        print(f"[{status}] True: {labels_true[i]} | Predicted: {labels_pred[i]}")
    
    # Save the model
    model_path = os.path.join(config.OUTPUT_DIR, "LSC_MVP_model.keras")
    model.save(model_path)
    logging.info(f"LSTM Model architecture and weights exported to {model_path}")
    
    # Save the LabelEncoder
    le_path = os.path.join(config.OUTPUT_DIR, "label_encoder.pkl")
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logging.info(f"LabelEncoder dictionary exported to {le_path}")
    
    logging.info("MVP Training & Export Complete.")

if __name__ == "__main__":
    train_lstm_mvp()
