import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
from shapely.geometry import Point # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report # type: ignore
import joblib # type: ignore

# === STEP 1: Load Labeled Field Data ===
# Assumed format: CSV with lat, lon, label (0=crop, 1=weed), plus optional metadata
csv_path = "data/labeled_crop_weed_samples.csv"
df = pd.read_csv(csv_path)

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# === STEP 2: Extract Features from Sentinel-2 or Earth Engine ===
# For now, assume spectral bands already exist in the CSV (e.g., B2, B3, ..., B12)
features = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
X = df[features]
y = df["label"]

# === STEP 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === STEP 4: Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === STEP 5: Evaluate Model ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Crop", "Weed"]))

# === STEP 6: Save Model ===
model_path = "models/crop_classifier_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")
