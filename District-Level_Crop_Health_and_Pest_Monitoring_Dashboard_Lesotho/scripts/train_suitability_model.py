import os
import numpy as np  # type: ignore
import rasterio  # type: ignore
import tensorflow as tf  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils.class_weight import compute_class_weight  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
import joblib  # type: ignore

def load_rasters(raster_paths):
    rasters = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            rasters.append(src.read(1))
    return np.stack(rasters, axis=-1)

def train_and_predict(crop_name, label_path, raster_paths):
    print(f"\\n--- Processing {crop_name} ---")

    x = load_rasters(raster_paths)

    with rasterio.open(label_path) as label_src:
        y = label_src.read(1)
        label_meta = label_src.meta

    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.flatten()

    valid_mask = (y_flat >= 0) & ~np.isnan(y_flat) & np.all(~np.isnan(x_flat), axis=1)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_valid)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_valid, test_size=0.2, random_state=42)

    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights))

    n_classes = len(unique_classes)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=256,
              validation_split=0.1,
              class_weight=class_weight_dict,
              verbose=1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{crop_name} Test Accuracy: {acc:.4f}")

    y_pred_test = np.argmax(model.predict(x_test), axis=1)
    print(classification_report(y_test, y_pred_test))

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{crop_name}_model.h5")
    joblib.dump(scaler, f"models/{crop_name}_scaler.pkl")
    print(f"Model and scaler saved for {crop_name}")

if __name__ == "__main__":
    crop = "potato"
    label_tif = "rasters/potato_labels.tif"
    feature_rasters = [
        "rasters/soil_moisture.tif",
        "rasters/precipitation.tif",
        "rasters/temperature.tif",
    ]
    train_and_predict(crop, label_tif, feature_rasters)
