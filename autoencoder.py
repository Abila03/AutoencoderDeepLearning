import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

# Konfigurasi
IMG_SIZE = (128, 128)
INPUT_FOLDER = "dataset/input"
OUTPUT_FOLDER = "dataset/output"
SOURCE_FOLDER = "source_images"
NUM_IMAGES = 30

# Buat gambar dummy
def buat_dummy_images():
    os.makedirs(SOURCE_FOLDER, exist_ok=True)
    for i in range(NUM_IMAGES):
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        img = np.full((IMG_SIZE[1], IMG_SIZE[0], 3), color, dtype=np.uint8)
        cv2.putText(img, f"{i}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imwrite(f"{SOURCE_FOLDER}/img_{i}.jpg", img)
    print(f"{NUM_IMAGES} gambar dummy berhasil dibuat di folder '{SOURCE_FOLDER}'")

# Siapkan dataset
def prepare_dataset():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Mempersiapkan dataset...")
    count = 0
    for i, file in enumerate(sorted(os.listdir(SOURCE_FOLDER))):
        img_path = os.path.join(SOURCE_FOLDER, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Gagal memuat gambar: {file}")
            continue
        img = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(f"{INPUT_FOLDER}/{i}.jpg", gray_rgb)
        cv2.imwrite(f"{OUTPUT_FOLDER}/{i}.jpg", img)
        count += 1
    print(f"Dataset siap! Total gambar diproses: {count}")

# Load dataset
def load_dataset(input_folder, output_folder):
    X, Y = [], []
    for file in sorted(os.listdir(input_folder)):
        input_img = cv2.imread(os.path.join(input_folder, file))
        output_img = cv2.imread(os.path.join(output_folder, file))
        if input_img is None or output_img is None:
            continue
        X.append(input_img / 255.0)
        Y.append(output_img / 255.0)
    return np.array(X), np.array(Y)

# Bangun model autoencoder
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Visualisasi hasil
def show_results(model, X_test, Y_test, num_samples=5):
    preds = model.predict(X_test)
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(X_test[i])
        axes[0].set_title("Input (Grayscale RGB)")
        axes[1].imshow(Y_test[i])
        axes[1].set_title("Output Asli")
        axes[2].imshow(preds[i])
        axes[2].set_title("Rekonstruksi")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.show()

# Eksekusi utama
if __name__ == "__main__":
    buat_dummy_images()
    prepare_dataset()
    X, Y = load_dataset(INPUT_FOLDER, OUTPUT_FOLDER)

    print(f"Jumlah data yang dimuat: {len(X)}")

    if len(X) == 0:
        print("❌ Gagal: Tidak ada data ditemukan. Pastikan folder gambar tidak kosong.")
        exit()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = build_autoencoder(X_train.shape[1:])
    print("Melatih model...")
    model.fit(X_train, Y_train, epochs=50, batch_size=5, validation_data=(X_test, Y_test))

    print("Menampilkan hasil prediksi...")
    show_results(model, X_test, Y_test)
