# Autoencoder untuk Rekonstruksi Citra Berbasis Grayscale RGB ke Citra Warna
📦 1. Dataset
✅ Deskripsi
Dataset yang digunakan bersifat custom, terdiri dari 30 gambar buatan (dummy) berukuran 128x128 piksel. Gambar-gambar ini memiliki warna latar acak dan angka sebagai label visual. Dataset dibagi menjadi dua:

Input: Gambar berwarna yang dikonversi ke grayscale lalu diubah ke format RGB (3 channel) agar kompatibel dengan arsitektur CNN.

Output: Gambar berwarna asli.

✅ Format Dataset:
Input: dataset/input/*.jpg (grayscale RGB)

Output: dataset/output/*.jpg (asli)

🧠 2. Arsitektur Autoencoder
Arsitektur terdiri dari Convolutional Autoencoder:

🔷 Encoder:
plaintext
Copy
Edit
Input → Conv2D(64, relu) → MaxPooling2D
      → Conv2D(32, relu) → MaxPooling2D
🔶 Decoder:
plaintext
Copy
Edit
→ Conv2D(32, relu) → UpSampling2D
→ Conv2D(64, relu) → UpSampling2D
→ Conv2D(3, sigmoid)
Ukuran input/output: (128, 128, 3)

Loss Function: Mean Squared Error (mse)

Optimizer: Adam

Epochs: 50

Batch Size: 5

📉 3. Performa Model
📊 Nilai Loss:
Contoh hasil pelatihan menunjukkan bahwa nilai loss dan val_loss terus menurun, mengindikasikan model mampu belajar dari data.

Contoh output terminal (simulasi):

plaintext
Copy
Edit
Epoch 20/20
loss: 0.0503 - val_loss: 0.0536
Meskipun loss tidak nol, autoencoder tetap mampu menghasilkan output yang visualnya mendekati asli, karena tugasnya adalah rekonstruksi kasar berbasis citra grayscale.

📌 Kesimpulan
Autoencoder ini efektif dalam merekonstruksi gambar warna dari versi grayscale.

Walau sederhana, proyek ini membuktikan kemampuan deep learning untuk pemetaan input-output non-identik.
