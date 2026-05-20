from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "chest_xray"
MODEL_PATH = BASE_DIR / "pneumonia_model.keras"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
EVALUATION_REPORT_PATH = BASE_DIR / "evaluation_report.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42
VALIDATION_SPLIT = 0.2
PREDICTION_THRESHOLD = 0.8
THRESHOLDS_TO_EVALUATE = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset_path():
    required_dirs = [
        DATASET_DIR / "train" / "NORMAL",
        DATASET_DIR / "train" / "PNEUMONIA",
        DATASET_DIR / "test" / "NORMAL",
        DATASET_DIR / "test" / "PNEUMONIA",
    ]
    missing_dirs = [path for path in required_dirs if not path.exists()]

    if missing_dirs:
        missing = "\n".join(f"- {path}" for path in missing_dirs)
        raise FileNotFoundError(
            "Struktur dataset belum lengkap. Pastikan dataset Kaggle sudah "
            f"diekstrak ke {DATASET_DIR}.\nFolder yang belum ditemukan:\n{missing}"
        )

    empty_dirs = []
    for path in required_dirs:
        image_count = sum(
            1 for file_path in path.rglob("*") if file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count == 0:
            empty_dirs.append(path)

    if empty_dirs:
        empty = "\n".join(f"- {path}" for path in empty_dirs)
        raise FileNotFoundError(
            "Folder dataset sudah ada, tetapi belum berisi gambar X-Ray. "
            "Download dataset dari Kaggle, lalu salin file gambar ke struktur "
            f"dataset yang benar.\nFolder kosong:\n{empty}"
        )


def count_images(directory):
    return sum(
        1 for file_path in directory.rglob("*") if file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def compute_class_weights():
    class_counts = {
        index: count_images(DATASET_DIR / "train" / class_name)
        for index, class_name in enumerate(CLASS_NAMES)
    }
    total_images = sum(class_counts.values())
    class_weights = {
        index: total_images / (len(CLASS_NAMES) * count)
        for index, count in class_counts.items()
    }

    print("\nClass counts")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {class_counts[index]}")

    print("\nClass weights")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {class_weights[index]:.4f}")

    return class_weights


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "train",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / "test",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds


def build_model():
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="pneumonia_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def plot_confusion_matrix(y_true, y_pred, threshold):
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    return matrix


def evaluate_model(model, test_ds):
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds]).astype(int)
    y_prob = model.predict(test_ds, verbose=1).ravel()

    report_lines = ["Threshold Comparison"]
    for threshold in THRESHOLDS_TO_EVALUATE:
        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        report_lines.append(f"\nThreshold: {threshold}")
        report_lines.append(report)

    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    matrix = plot_confusion_matrix(y_true, y_pred, PREDICTION_THRESHOLD)
    final_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report_lines.extend(
        [
            f"\nFinal Evaluation Results (threshold={PREDICTION_THRESHOLD})",
            f"Accuracy : {accuracy:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall   : {recall:.4f}",
            f"F1-score : {f1:.4f}",
            "\nConfusion Matrix",
            str(matrix),
            "\nClassification Report",
            final_report,
        ]
    )

    report_text = "\n".join(report_lines)
    EVALUATION_REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Evaluation report saved to: {EVALUATION_REPORT_PATH}")


def main():
    validate_dataset_path()
    class_weights = compute_class_weights()
    train_ds, val_ds, test_ds = load_datasets()
    model = build_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.summary()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    if not MODEL_PATH.exists():
        model.save(MODEL_PATH)

    best_model = tf.keras.models.load_model(MODEL_PATH)
    evaluate_model(best_model, test_ds)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
