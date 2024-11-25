import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# Constants
latent_dim = 100
IMG_SIZE = 150
batch_size = 64
epochs = 50
fine_tune_epochs = 30
train_dir = '/home/saivarun/AD/balanced_dataset/train'
validation_dir = '/home/saivarun/AD/balanced_dataset/test'

# Define GAN Generators
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(19*19*256, input_dim=latent_dim))
    model.add(layers.Reshape((19, 19, 256)))
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding="same", activation="tanh"))
    model.add(layers.Resizing(IMG_SIZE, IMG_SIZE))  # Resize to 150x150
    return model

# Define GAN Discriminator
def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

# Compile the discriminator
img_shape = (IMG_SIZE, IMG_SIZE, 3)
generator = build_generator()
discriminator = build_discriminator(img_shape)
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-5), metrics=["accuracy"])

# GAN Model (Combine Generator and Discriminator)
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = models.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-5))

# Set up ImageDataGenerator for Classification
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build and compile the EfficientNet model for classification
base_model = EfficientNetB3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(train_data.num_classes, activation="softmax")
])
model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the Classification Model
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs
)

# GAN Training Loop (Separate from the classification model)
def train_gan(epochs, batch_size=64):
    for epoch in range(epochs):
        # Train discriminator with real images
        real_images, _ = train_data.next()
        real_labels = np.ones((batch_size, 1))  # Real labels (1)

        # Train discriminator with fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # Latent space for fake images
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))  # Fake labels (0)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        # Train generator (through GAN, with the discriminator frozen)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # Random noise for generator
        g_loss = gan.train_on_batch(noise, real_labels)  # We want the generator to fool the discriminator

        # Print losses
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss[0]}")

# Train GAN separately
train_gan(epochs=1000)

# After GAN training, you can proceed to fine-tuning the classification model if needed
base_model.trainable = True
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=fine_tune_epochs
)

# Feature Extraction for Decision Trees
def extract_features(model, data):
    features = []
    for i in range(len(data)):
        image = np.expand_dims(data[i][0], axis=0)  # Expand the dimensions to simulate batch
        feature = model.predict(image)
        features.append(feature.flatten())  # Flatten the feature for Decision Tree
    return np.array(features)

# Extract features for training Decision Tree
train_features = extract_features(base_model, train_data)
validation_features = extract_features(base_model, validation_data)

# Normalize features before applying PCA
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
validation_features_scaled = scaler.transform(validation_features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)  # Reduce to 100 components (tune this value)
train_features_pca = pca.fit_transform(train_features_scaled)
validation_features_pca = pca.transform(validation_features_scaled)

# Grid search for hyperparameter tuning of Decision Tree
param_grid = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(train_features_pca, train_data.classes)

print(f"Best Decision Tree Parameters: {grid_search.best_params_}")

# Train with the best parameters found
best_dt = grid_search.best_estimator_
best_dt.fit(train_features_pca, train_data.classes)

# Evaluate the tuned Decision Tree
dt_predictions = best_dt.predict(validation_features_pca)
dt_accuracy = accuracy_score(validation_data.classes, dt_predictions)
print(f"Decision Tree Accuracy after Hyperparameter Tuning: {dt_accuracy * 100:.2f}%")

# Ensemble Learning with Weighted Voting
def ensemble_predictions_weighted(model, dt_classifier, features, ef_weight=0.7, dt_weight=0.3):
    ef_predictions = model.predict(features)
    dt_predictions = dt_classifier.predict(features)
    
    ensemble_preds = []
    for i in range(len(ef_predictions)):
        ef_pred = np.argmax(ef_predictions[i])
        dt_pred = dt_predictions[i]
        
        if ef_pred == dt_pred:
            ensemble_preds.append(ef_pred)
        else:
            weighted_pred = ef_weight * ef_pred + dt_weight * dt_pred
            ensemble_preds.append(np.round(weighted_pred).astype(int))
    
    return np.array(ensemble_preds)

# Apply ensemble on validation data
ensemble_preds = ensemble_predictions_weighted(model, best_dt, validation_features_pca)

# Evaluate ensemble performance
ensemble_accuracy = accuracy_score(validation_data.classes, ensemble_preds)
print(f"Ensemble Accuracy after Weighted Voting: {ensemble_accuracy * 100:.2f}%")
