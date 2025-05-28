import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class SimpleCNNClassifier:
    def __init__(self, num_classes=None, input_size=(32, 32, 3)):
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = None

    def preprocess_image(self, image_path, target_size=(32, 32)):
        """
        Load and preprocess a single image to 32x32 pixels
        """
        try:
            # Load image
            img = load_img(image_path)

            # Convert to array
            img_array = img_to_array(img)

            # Resize to 32x32
            img_resized = cv2.resize(img_array, target_size)

            # Ensure 3 channels (RGB)
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            elif img_resized.shape[2] == 1:
                img_resized = np.repeat(img_resized, 3, axis=2)

            return img_resized

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def load_metadata(self, metadata_path):
        """
        Load metadata CSV file and return dataframe
        Expected columns: filename, class (or similar)
        """
        try:
            df = pd.read_csv(metadata_path)
            print(f"Loaded metadata with shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample data:")
            print(df.head())
            return df
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None

    def load_and_preprocess_dataset(self, image_paths, labels):
        """
        Load and preprocess entire dataset
        """
        processed_images = []
        valid_labels = []

        print(f"Processing {len(image_paths)} images...")

        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)} images")

            processed_img = self.preprocess_image(img_path)

            if processed_img is not None:
                processed_images.append(processed_img)
                valid_labels.append(label)

        # Convert to numpy arrays
        X = np.array(processed_images, dtype=np.float32)
        y = np.array(valid_labels)

        # Normalize pixel values to [0, 1]
        X = X / 255.0

        print(f"Final dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        return X, y

    def load_and_preprocess_test_images(self, image_paths):
        """
        Load and preprocess test images (no labels)
        """
        processed_images = []
        valid_paths = []

        print(f"Processing {len(image_paths)} test images...")

        for i, img_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)} test images")

            processed_img = self.preprocess_image(img_path)

            if processed_img is not None:
                processed_images.append(processed_img)
                valid_paths.append(img_path)

        # Convert to numpy arrays
        X = np.array(processed_images, dtype=np.float32)

        # Normalize pixel values to [0, 1]
        X = X / 255.0

        print(f"Final test dataset shape: {X.shape}")

        return X, valid_paths

    def prepare_dataset_from_folder(
        self, images_folder, metadata_path, filename_col="filename", class_col="class"
    ):
        """
        Prepare dataset from folder structure with CSV metadata
        """
        # Load metadata
        df = self.load_metadata(metadata_path)
        if df is None:
            return None, None

        # Drop id column if it exists (but keep it for test data)
        df_working = df.copy()
        if "id" in df_working.columns and class_col in df_working.columns:
            df_working = df_working.drop(columns=["id"])

        # Check if specified columns exist
        if filename_col not in df_working.columns:
            print(
                f"Column '{filename_col}' not found. Available columns: {df_working.columns.tolist()}"
            )
            return None, None

        if class_col not in df_working.columns:
            print(
                f"Column '{class_col}' not found. Available columns: {df_working.columns.tolist()}"
            )
            return None, None

        # Get unique classes and encode labels
        self.class_names = sorted(df_working[class_col].unique())
        self.num_classes = len(self.class_names)
        print(f"Found {self.num_classes} unique classes: {self.class_names}")

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(df_working[class_col])

        # Create full image paths
        image_paths = []
        labels = []
        missing_files = []

        for idx, row in df_working.iterrows():
            img_path = os.path.join(images_folder, row[filename_col])

            # Check if file exists
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(encoded_labels[idx])
            else:
                missing_files.append(row[filename_col])

        if missing_files:
            print(f"Warning: {len(missing_files)} files not found in {images_folder}")
            if len(missing_files) <= 10:
                print("Missing files:", missing_files)

        print(f"Found {len(image_paths)} valid image files")

        # Load and preprocess images
        X, y = self.load_and_preprocess_dataset(image_paths, labels)

        return X, y

    def prepare_test_dataset(
        self, images_folder, metadata_path, filename_col="image_path"
    ):
        """
        Prepare test dataset from folder structure with CSV metadata (no labels)
        """
        # Load metadata
        df = self.load_metadata(metadata_path)
        if df is None:
            return None, None

        # Check if specified columns exist
        if filename_col not in df.columns:
            print(
                f"Column '{filename_col}' not found. Available columns: {df.columns.tolist()}"
            )
            return None, None

        if "id" not in df.columns:
            print("Column 'id' not found in test metadata")
            return None, None

        # Create full image paths
        image_paths = []
        ids = []
        missing_files = []

        for idx, row in df.iterrows():
            img_path = os.path.join(images_folder, row[filename_col])

            # Check if file exists
            if os.path.exists(img_path):
                image_paths.append(img_path)
                ids.append(row["id"])
            else:
                missing_files.append(row[filename_col])

        if missing_files:
            print(
                f"Warning: {len(missing_files)} test files not found in {images_folder}"
            )
            if len(missing_files) <= 10:
                print("Missing test files:", missing_files)

        print(f"Found {len(image_paths)} valid test image files")

        # Load and preprocess images
        X, valid_paths = self.load_and_preprocess_test_images(image_paths)

        return X, ids

    def get_class_distribution(self, y):
        """
        Print class distribution
        """
        if self.class_names is not None:
            unique, counts = np.unique(y, return_counts=True)
            print("\nClass Distribution:")
            for class_idx, count in zip(unique, counts):
                class_name = self.class_names[class_idx]
                print(f"Class {class_idx} ({class_name}): {count} samples")
        else:
            unique, counts = np.unique(y, return_counts=True)
            print("\nClass Distribution:")
            for class_idx, count in zip(unique, counts):
                print(f"Class {class_idx}: {count} samples")

    def create_model(self):
        """
        Create a simple CNN model optimized for 32x32 input
        """
        inputs = tf.keras.Input(shape=self.input_size)

        # First convolutional block
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Second convolutional block
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Third convolutional block
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Fourth convolutional block
        x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Flatten and dense layers
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)

        # Output layer
        predictions = Dense(self.num_classes, activation="softmax")(x)

        # Create the model
        self.model = Model(inputs=inputs, outputs=predictions)

        return self.model

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("Model compiled successfully!")

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
            ),
        ]

        # Train the model
        history = self.model.fit(
            X_train,
            y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        return test_loss, test_accuracy

    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes, predictions

    def predict_test_data(self, X_test, ids, output_filename="submission.csv"):
        """
        Make predictions on test data and save to CSV
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        print("\nMaking predictions on test data...")
        predictions = self.model.predict(X_test, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # Create submission dataframe
        submission_df = pd.DataFrame({"id": ids, "ClassId": predicted_classes})

        # Sort by id to ensure consistent ordering
        submission_df = submission_df.sort_values("id")

        # Save to CSV
        submission_df.to_csv(output_filename, index=False)
        print(f"Predictions saved to {output_filename}")
        print(f"Number of predictions: {len(submission_df)}")
        print(f"Sample predictions:")
        print(submission_df.head())

        return submission_df

    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(history.history["accuracy"], label="Training Accuracy")
        ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # Plot loss
        ax2.plot(history.history["loss"], label="Training Loss")
        ax2.plot(history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# Example usage with your folder structure:
if __name__ == "__main__":
    # Initialize classifier
    classifier = SimpleCNNClassifier()

    # Set your paths
    images_folder = "train/Images"
    metadata_path = "train/train_metadata.csv"

    # Load and prepare dataset
    print("Loading dataset from folder and CSV...")
    X, y = classifier.prepare_dataset_from_folder(
        images_folder=images_folder,
        metadata_path=metadata_path,
        filename_col="image_path",  # Change this to match your CSV column name
        class_col="ClassId",  # Change this to match your CSV column name
    )

    if X is not None and y is not None:
        # Show class distribution
        classifier.get_class_distribution(y)

        # Split data
        print("\nSplitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Create and compile model
        print("\nCreating model...")
        model = classifier.create_model()
        classifier.compile_model(learning_rate=0.001)

        print(f"Model created with {classifier.num_classes} classes")
        print("Model summary:")
        model.summary()

        # Train model
        print("\nStarting training...")
        history = classifier.train_model(X_train, y_train, X_val, y_val, epochs=50)

        # Plot training history
        classifier.plot_training_history(history)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = classifier.evaluate_model(X_test, y_test)

        # Make predictions on test set
        predictions, probabilities = classifier.predict(X_test)

        print(f"\nFinal Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Save the model
        classifier.save_model("traffic_sign_cnn_model.h5")

        # Generate classification report
        from sklearn.metrics import classification_report, confusion_matrix

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Plot confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

        # NOW PREDICT ON TEST DATA
        print("\n" + "=" * 50)
        print("PREDICTING ON TEST DATA")
        print("=" * 50)

        # Load test data
        test_images_folder = "test/Images"
        test_metadata_path = "test/test_metadata.csv"

        print("Loading test dataset...")
        X_test_final, test_ids = classifier.prepare_test_dataset(
            images_folder=test_images_folder,
            metadata_path=test_metadata_path,
            filename_col="image_path",
        )

        if X_test_final is not None and test_ids is not None:
            # Make predictions and save to CSV
            submission_df = classifier.predict_test_data(
                X_test_final, test_ids, output_filename="final_submission.csv"
            )

            print(f"\nSubmission file created successfully!")
            print(f"Contains {len(submission_df)} predictions")

        else:
            print("Failed to load test data")

    else:
        print("Failed to load dataset. Please check your file paths and CSV format.")
        print("\nExpected CSV format:")
        print("filename,class")
        print("image1.jpg,class_name_1")
        print("image2.jpg,class_name_2")
        print("...")

        print(f"\nMake sure:")
        print(f"1. Images are in: {images_folder}")
        print(f"2. CSV file is at: {metadata_path}")
        print(f"3. CSV has columns for filename and class")

    print("\nDone!")
