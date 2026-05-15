import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from src.config import (MAX_SEQUENCE_LENGTH, CNN_FILTERS, CNN_KERNEL_SIZE, LSTM_UNITS, 
                        DENSE_UNITS, DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, EPOCHS, 
                        EARLY_STOPPING_PATIENCE, MODELS_DIR)

def build_cnn_lstm_model(input_shape):
    print("\n" + "=" * 50)
    print("Building CNN-LSTM Hybrid Model")
    print("=" * 50)
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    x = layers.Conv2D(filters=CNN_FILTERS, kernel_size=(CNN_KERNEL_SIZE, 1), activation='relu', padding='same', name='conv2d_1')(x)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name='maxpool_1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_1')(x)
    
    x = layers.Conv2D(filters=CNN_FILTERS * 2, kernel_size=(CNN_KERNEL_SIZE, 1), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='batch_norm_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), name='maxpool_2')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_2')(x)
    
    x = layers.Reshape((-1, CNN_FILTERS * 2))(x)
    
    x = layers.LSTM(LSTM_UNITS, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name='lstm_1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_3')(x)
    
    x = layers.LSTM(LSTM_UNITS // 2, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name='lstm_2')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_4')(x)
    
    for i, units in enumerate(DENSE_UNITS):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        x = layers.Dropout(DROPOUT_RATE / 2, name=f'dropout_dense_{i+1}')(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    
    return model

def create_sequences_for_lstm(X, y, sequence_length):
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1
    
    X_sequences = np.zeros((n_sequences, sequence_length, n_features))
    y_sequences = np.zeros(n_sequences)
    
    for i in range(n_sequences):
        X_sequences[i] = X[i:i + sequence_length]
        y_sequences[i] = np.max(y[i:i + sequence_length])
    
    return X_sequences, y_sequences

def train_cnn_lstm(model, X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCHS):
    print("\n" + "=" * 50)
    print("Training CNN-LSTM Model")
    print("=" * 50)
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(filepath=f"{MODELS_DIR}/cnn_lstm_best.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

def evaluate_cnn_lstm(model, X_test, y_test):
    print("\n" + "=" * 50)
    print("Evaluating CNN-LSTM Model")
    print("=" * 50)
    
    loss, accuracy, precision, recall, auc = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    from sklearn.metrics import confusion_matrix, f1_score
    
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    print(f"\nCNN-LSTM Test Performance:")
    print(f"  Loss:      {loss:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    return metrics