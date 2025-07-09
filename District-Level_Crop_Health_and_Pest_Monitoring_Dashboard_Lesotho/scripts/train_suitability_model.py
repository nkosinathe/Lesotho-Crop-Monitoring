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

# -----------------------
# Function to Load and Stack Raster Features
# -----------------------

def load_rasters(raster_paths):
    rasters = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            rasters.append(src.read(1))
    return np.stack(rasters, axis=-1)

# -----------------------
# Function to Train, Predict, and Save Map + Model
# -----------------------

def train_and_predict(crop_name, label_path, raster_paths):
    print(f"\\n--- Processing {crop_name} ---")
    
    # Load input features
    x = load_rasters(raster_paths)
    
    # Load suitability labels
    with rasterio.open(label_path) as label_src:
        y = label_src.read(1)
        label_meta = label_src.meta
        
    # Prepare dataset
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.flatten()
    
    # Filter valid data
    valid_mask = (y_flat >= 0) & ~np.isnan(y_flat) & np.all(~np.isnan(x_flat), axis=1)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]

    # Normalize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_valid)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_valid, test_size=0.2, random_state=42
    )

    # Handle class imbalance
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weight_dict = dict(zip(unique_classes, class_weights))

    # Model architecture
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

    # Evaluation
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{crop_name} Test Accuracy: {acc:.4f}")

    y_pred_test = np.argmax(model.predict(x_test), axis=1)
    print(classification_report(y_test, y_pred_test))
    
    # Predict full raster
    x_all_scaled = scaler.transform(x_flat)
    predictions = model.predict(x_all_scaled)
    predicted_labels = np.argmax(predictions, axis=1)

    # Rebuild full raster shape
    output_array = np.full(y_flat.shape, -1)
    output_array[valid_mask] = predicted_labels
    output_raster = output_array.reshape(y.shape)

    # Create outputs folder
    os.makedirs("models", exist_ok=True)
    
    # Save predicted raster
    output_raster_path = os.path.join("outputs", f'predicted_suitability_{crop_name.lower().replace(" ", "_")}.tif')
    with rasterio.open(output_raster_path, 'w',
                       driver='GTiff',
                       height=output_raster.shape[0],
                       width=output_raster.shape[1],
                       count=1,
                       dtype=rasterio.uint8,
                       crs=label_meta['crs'],
                       transform=label_meta['transform']) as dst:
        dst.write(output_raster.astype(np.uint8), 1)

    print(f"{crop_name} suitability map saved: {output_raster_path}")

    # Save model
    model_path = os.path.join("outputs", f"{crop_name.lower().replace(' ', '_')}_model.h5")
    model.save(model_path)
    print(f"{crop_name} model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join("outputs", f"{crop_name.lower().replace(' ', '_')}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"{crop_name} scaler saved: {scaler_path}")
    
# -----------------------
# File Paths Setup
# -----------------------

input_rasters = [
    'soil_moisture.tif',
    'soil_texture.tif',
    'land_use_class.tif',
    'slope.tif',
    'temperature.tif',
    'aspect.tif',
    'elevation.tif',
    'rainfall.tif',
    'organic_carbon.tif',
    'soil_ph.tif'
]

crops = {
    'Potatoes': 'suitability_potatoes.tif',
    'Apples': 'suitability_apples.tif',
    'Sweet Potatoes': 'suitability_sweet_potatoes.tif'
}

# -----------------------
# Run For All Crops
# -----------------------

for crop_name, label_path in crops.items():
    train_and_predict(crop_name, label_path, input_rasters)