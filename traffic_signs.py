import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
EPOCHS = 15
BATCH_SIZE = 32


def get_model():
    model = Sequential()


    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())


    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))


    model.add(Dense(NUM_CATEGORIES, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():

    if len(sys.argv) < 2:
        sys.exit("Usage: python traffic_signs.py Train [model.h5]")

    data_dir = sys.argv[1]

    print("Loading data with generator...")


    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.4
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    print("Training model...")

    model = get_model()


    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stop]
    )

    print("Evaluating model...")
    loss, accuracy = model.evaluate(validation_generator, verbose=2)
    print(f"Accuracy: {accuracy:.4f}")


    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}")


if __name__ == "__main__":
    main()