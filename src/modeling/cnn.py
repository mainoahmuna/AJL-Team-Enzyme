# %%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import tensorflow.keras as keras
import time

# %%
train_path = '/Users/mainoahmuna/Downloads/Projects/AJL-Team-Enzyme/data/processed/train_split.csv'
train_image_dir = '/Users/mainoahmuna/Downloads/Projects/AJL-Team-Enzyme/data/raw/train_images'
test_path = '/Users/mainoahmuna/Downloads/Projects/AJL-Team-Enzyme/data/raw/test_images/'
test_csv = '/Users/mainoahmuna/Downloads/Projects/AJL-Team-Enzyme/data/splits/test_split.csv'

# %%
train_df = pd.read_csv(train_path)

# %%
train_df['label'].value_counts()

# %%
# Load dataset (assuming df is your DataFrame with a 'label' column)
# df = pd.read_csv("your_dataset.csv")

# Get class distribution
class_counts = train_df['label'].value_counts()
median_samples = class_counts.median()  # Use median as a balancing point

# Initialize an empty list to store resampled data
balanced_data = []

for label, count in class_counts.items():
    df_subset = train_df[train_df['label'] == label]
    
    if count > median_samples:
        # **Downsample majority classes**
        df_resampled = resample(df_subset, 
                                replace=False,  # No replacement (subsampling)
                                n_samples=int(median_samples),  
                                random_state=42)
    else:
        # **Upsample minority classes**
        df_resampled = resample(df_subset, 
                                replace=True,  # Oversample with replacement
                                n_samples=int(median_samples),  
                                random_state=42)

    balanced_data.append(df_resampled)

# Combine all resampled data
df_balanced = pd.concat(balanced_data)

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check new class distribution
print(df_balanced['label'].value_counts())

# Save or use the balanced dataset
# df_balanced.to_csv("balanced_dataset.csv", index=False)

# %%
# Combine label and md5hash to form the correct path
df_balanced['file_path'] = df_balanced['label'] + '/' + df_balanced['md5hash']

# %%
label_encoder = LabelEncoder()
df_balanced['encoded_label'] = label_encoder.fit_transform(df_balanced['label']).astype(str)

# %%
# Split the data into training and validation sets
train_data, val_data = train_test_split(df_balanced, test_size=0.2, random_state=42)

# %%
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# %%
# Apply data augmentation to training data
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values
    rotation_range=30,      # Random rotation up to 30 degrees
    width_shift_range=0.2,  # Shift image horizontally
    height_shift_range=0.2, # Shift image vertically
    shear_range=0.2,        # Apply shear transformations
    zoom_range=0.2,         # Zoom in/out randomly
    horizontal_flip=True,   # Flip images horizontally
    fill_mode='nearest'     # Fill missing pixels
)

# No augmentation for validation data
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Function remains the same
def create_generator(dataframe, directory, batch_size=32, target_size=(128, 128), is_train=True):
    datagen = train_datagen if is_train else val_datagen  # Use appropriate generator
    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col='file_path',  
        y_col='encoded_label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        validate_filenames=False  
    )
    return generator

# %%
train_generator = create_generator(train_data, train_image_dir)
val_generator = create_generator(val_data, train_image_dir)

# %%
val_data.drop(columns=['label'], inplace=True)
train_data.drop(columns=['label'], inplace=True)

# %%
model = keras.models.Sequential([
    # **First Convolutional Block**
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),  # Regularization

    # **Second Convolutional Block**
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    # **Third Convolutional Block**
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),

    # **Fourth Convolutional Block**
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.4),

    # **Fifth Convolutional Block**
    keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.5),

    # **Global Average Pooling + Dense Layers**
    keras.layers.GlobalAveragePooling2D(),  # Reduces parameters
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Multi-class classification
])

model.summary()

# %%
loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# %%
class ProgBarLoggerNEpochs(keras.callbacks.Callback):
    
    def __init__(self, num_epochs: int, every_n: int = 50):
        self.num_epochs = num_epochs
        self.every_n = every_n
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            s = 'Epoch [{}/ {}]'.format(epoch + 1, self.num_epochs)
            logs_s = ['{}: {:.4f}'.format(k.capitalize(), v)
                      for k, v in logs.items()]
            s_list = [s] + logs_s
            print(', '.join(s_list))

# %%
X_train = train_data.drop(columns=['encoded_label', 'file_path'])
y_train = train_data['encoded_label']
X_val = val_data.drop(columns=['encoded_label', 'file_path'])
y_val = val_data['encoded_label']

# %%
model.fit(
    train_generator,
    epochs=200,
    validation_data=val_generator
)

# %%
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

def create_test_generator(dataframe, directory, batch_size=32, target_size=(128, 128)):
    generator = test_datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col='file_path',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,  # No labels for test set
        shuffle=False  # Keep order for matching predictions
    )
    return generator

# %%
test_df = pd.read_csv(test_csv)
test_df['file_path'] = test_path + '/' + test_df['md5hash'] + '.jpg' # If needed

test_generator = create_test_generator(test_df, test_path)

# %%
predictions = model.predict(test_generator)

# %%
predicted_labels = predictions.argmax(axis=1)

# %%
test_df['label'] = label_encoder.inverse_transform(predicted_labels)

# %%
# Count occurrences of each label
label_counts = test_df['label'].value_counts(normalize=True) * 100
label_summary = label_counts.to_dict()

# Prepare the summary output
summary_output = f"md5hash\n\nlabel\n{len(test_df)}\n\nunique values\n"
for label, percentage in label_summary.items():
    summary_output += f"{label}\n{percentage:.0f}%\n"

summary_output += f"Other ({sum(label_counts < 5)}%)\n\n"

# Append actual predictions
output_df = test_df[['md5hash', 'label']]
output_df.to_csv('/Users/mainoahmuna/Downloads/Projects/AJL-Team-Enzyme/data/predictions/predictions.csv', index=False)

# Save summary to a text file
with open('/Users/mainoahmuna/Downloads/Projects/AJL-Team-Enzyme/data/predictions/summary.txt', 'w') as f:
    f.write(summary_output)

# %%



