import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision, regularizers
from tensorflow.keras.utils import Sequence

# ==========================================
# 1. HARDWARE CONFIGURATION
# ==========================================
def setup_hardware():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            mixed_precision.set_global_policy('mixed_float16')
            print(f"GPU Detected: {len(gpus)} device(s). Mixed Precision Enabled.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Training will be slow on CPU.")

# ==========================================
# 2. DATA GENERATOR
# ==========================================
class ImageGenerator(Sequence):
    def __init__(self, patient_ids, slope_map, batch_size, npy_dir, shuffle=True):
        self.patient_ids = patient_ids
        self.slope_map = slope_map
        self.batch_size = batch_size
        self.npy_dir = npy_dir
        self.shuffle = shuffle
        self.indices = np.arange(len(self.patient_ids))
        self.MAX_SLICES = 63

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.patient_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_ids = [self.patient_ids[k] for k in batch_indices]

        temp_images = []
        temp_slopes = []

        for pid in batch_ids:
            path = os.path.join(self.npy_dir, f"{pid}.npy")
            if not os.path.exists(path): continue

            try:
                img = np.load(path)
                depth = img.shape[0]
                if depth > self.MAX_SLICES:
                    start = (depth - self.MAX_SLICES) // 2
                    img = img[start : start + self.MAX_SLICES]

                if self.shuffle:
                    if np.random.rand() > 0.5: img = np.flip(img, axis=2)
                    if np.random.rand() > 0.5:
                        k = np.random.randint(1, 4)
                        img = np.rot90(img, k=k, axes=(1, 2))
                    h_shift = np.random.randint(-10, 10)
                    w_shift = np.random.randint(-10, 10)
                    img = scipy.ndimage.shift(img, (0, h_shift, w_shift, 0), order=0, mode='constant', cval=0)

                temp_images.append(img)
                temp_slopes.append(self.slope_map[pid])
            except Exception as e:
                print(f"Error loading {pid}: {e}")

        if len(temp_images) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        X_batch = np.zeros((len(temp_images), self.MAX_SLICES, 256, 256, 1), dtype=np.float32)
        for i, img in enumerate(temp_images):
            curr_slices = img.shape[0]
            offset = (self.MAX_SLICES - curr_slices) // 2
            X_batch[i, offset : offset + curr_slices, :, :, :] = img

        return X_batch, np.array(temp_slopes, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
def build_cnn_lstm_model():
    inp = layers.Input(shape=(None, 256, 256, 1))
    l2_reg = regularizers.l2(0.01)

    x = layers.TimeDistributed(layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=l2_reg))(inp)
    x = layers.TimeDistributed(layers.GroupNormalization(groups=4))(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=l2_reg))(x)
    x = layers.TimeDistributed(layers.GroupNormalization(groups=4))(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)

    x = layers.TimeDistributed(layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=l2_reg))(x)
    x = layers.TimeDistributed(layers.GroupNormalization(groups=4))(x)
    x = layers.TimeDistributed(layers.Activation('relu'))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.2))(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(0.5)
    output = layers.Dense(1, activation='linear', dtype='float32')(x)

    model = models.Model(inputs=inp, outputs=output, name="CNN_LSTM_Regressor")
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.Huber())
    return model

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM for Pulmonary Fibrosis")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing training_metadata.csv")
    parser.add_argument("--img_dir", type=str, default="./processed_train", help="Directory containing .npy images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (keep low for 3D data)")
    parser.add_argument("--results_dir", type=str, default="./results", help="Where to save models and plots")
    args = parser.parse_args()

    setup_hardware()
    os.makedirs(args.results_dir, exist_ok=True)

    # Load Metadata
    csv_path = os.path.join(args.data_dir, "training_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata not found at {csv_path}. Please run preprocessing first.")
    
    df = pd.read_csv(csv_path)
    slope_map = dict(zip(df.Patient, df.Slope))
    
    train_ids, val_ids = train_test_split(df.Patient.unique(), test_size=0.2, random_state=42)
    print(f"Training on {len(train_ids)} patients, Validating on {len(val_ids)} patients.")

    train_gen = ImageGenerator(train_ids, slope_map, args.batch_size, args.img_dir, shuffle=True)
    val_gen = ImageGenerator(val_ids, slope_map, args.batch_size, args.img_dir, shuffle=False)

    model = build_cnn_lstm_model()
    model.summary()

    checkpoint_path = os.path.join(args.results_dir, "best_cnn_lstm.keras")
    callbacks_list = [
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
        callbacks.EarlyStopping(patience=6, monitor="val_loss", restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print("\n--- Starting Training ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks_list
    )

    print("\n--- Saving Results ---")
    
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curves (Huber)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, "training_history.png"))
    
    y_true, y_pred = [], []
    for i in range(len(val_gen)):
        bx, by = val_gen[i]
        preds = model.predict(bx, verbose=0)
        y_true.extend(by)
        y_pred.extend(preds.flatten())
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"Final MAE: {mae:.4f}")
    print(f"Final RMSE: {rmse:.4f}")
    
    with open(os.path.join(args.results_dir, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}")

if __name__ == "__main__":
    main()