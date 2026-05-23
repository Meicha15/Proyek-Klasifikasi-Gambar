## Deskripsi Proyek
Klasifikasi gambar berbasis kecerdasan buatan (*Computer Vision*) memegang peranan krusial dalam berbagai sub-bidang teknologi modern, mulai dari otomatisasi pengelolaan satwa liar, diagnosis medis veteriner, hingga ekosistem *e-commerce* berbasis kebutuhan hewan peliharaan. Identifikasi taksonomi hewan secara manual sering kali tidak efisien, membutuhkan keahlian spesifik, dan rentan terhadap kesalahan manusia (*human error*), khususnya ketika menangani volume data visual yang masif.

Melalui arsitektur jaringan saraf tiruan konvensional atau *Convolutional Neural Network* (CNN), sistem komputer dilatih untuk mengekstrak fitur spasial hierarkis dari sebuah gambar seperti tekstur wajah, bentuk telinga, pola bulu, dan struktur hidung secara mandiri. Proyek ini memfokuskan pengembangan sistem klasifikasi multi-kelas untuk membedakan wajah tiga kategori hewan utama, yaitu Kucing (*Cat*), Anjing (*Dog*), dan Satwa Liar (*Wild*). Model yang andal dapat diintegrasikan ke dalam aplikasi seluler untuk membantu pemilik hewan peliharaan, dokter hewan, maupun peneliti konservasi alam dalam mengidentifikasi kelompok spesimen secara otomatis, akurat, dan aktual.

- Melakukan preprocessing gambar (resizing, normalisasi).
- Membangun model CNN (Convolutional Neural Network) menggunakan TensorFlow.
- Melatih model dengan dataset gambar.
- Mengevaluasi performa model dengan akurasi dan loss.
- Menyimpan model hasil pelatihan.

## Cara Menjalankan

1. Pastikan Python telah terinstall.
2. Install semua dependensi dengan perintah: pip install -r requirements.txt
3. Jalankan file notebook `Proyek_Klasifikasi_Gambar.ipynb` menggunakan Jupyter Notebook atau VSCode.

## Struktur Folder

- `dataset/` : Berisi dataset gambar.
- `models/` : Berisi model hasil pelatihan.
- `Proyek_Klasifikasi_Gambar.ipynb` : Notebook utama proyek.
- `requirements.txt` : Daftar library Python yang dibutuhkan.
- `README.md` : Dokumentasi proyek ini.

## Teknologi yang Digunakan

- Python
- TensorFlow
- Matplotlib
- NumPy
- Seaborn

## Business Understanding

### Problem Statements
1. Bagaimana merancang arsitektur model *Deep Learning* berbasis *Convolutional Neural Network* (CNN) yang efisien untuk membedakan tiga kelas gambar wajah hewan (*Cat*, *Dog*, *Wild*) secara akurat?
2. Bagaimana mengatasi tantangan variasi spasial dan ukuran piksel gambar yang beragam agar model dapat mengenali pola wajah secara konsisten?
3. Bagaimana menguji reliabilitas dan keterpisahan fitur model prediksi melalui evaluasi metrik klasifikasi komprehensif seperti *Confusion Matrix* dan *Classification Report*?

### Goals
1. Membangun model klasifikasi gambar multi-kelas berbasis arsitektur *Sequential* CNN kustom menggunakan *framework* TensorFlow/Keras.
2. Menerapkan teknik standardisasi resolusi input gambar dan augmentasi generator data gambar (*ImageDataGenerator*) untuk menjamin kestabilan performa model.
3. Mencapai akurasi pengujian (*test accuracy*) di atas 95% guna memenuhi kriteria klasifikasi yang reliabel untuk kebutuhan sistem inferensi real-time.

### Solution Statements
1. **Arsitektur CNN Kustom & Regularisasi**: Mengimplementasikan kombinasi lapisan konvolusi berganda (`Conv2D`), penormalan kelompok data (`BatchNormalization`), pencarian nilai maksimal spasial (`MaxPool2D`), serta pencegahan *overfitting* menggunakan lapisan pemutusan acak (`Dropout`).
2. **Mekanisme Otomatis Callback**: Menyematkan *Custom Callback* untuk menghentikan proses pembelajaran secara dinamis jika model telah menyentuh batas performa ideal (Akurasi $\ge 95\%$ dan Loss $\le 0.1$) guna menghemat konsumsi daya komputasi.
3. **Analisis Komprehensif Akhir**: Mengevaluasi performa model menggunakan subset data pengujian independen melalui analisis mendalam terhadap skor presisi, *recall*, *F1-score*, dan visualisasi matriks kesalahan (*confusion matrix*).

---

## Data Understanding
Dataset utama yang digunakan dalam proyek ini dinamakan **Animal Face Dataset**, yang merupakan kumpulan data citra wajah hewan yang terbagi ke dalam tiga direktori kelas utama: kucing, anjing, dan satwa liar (*wild*).

### Karakteristik Dataset:
* **Total Sampel Uji**: 3.226 citra pada subset evaluasi akhir.
* **Pembagian Kelas (Support)**:
  * **Cat**: 1.119 citra wajah kucing.
  * **Dog**: 1.057 citra wajah anjing.
  * **Wild**: 1.050 citra wajah satwa liar.
* **Format Data**: Citra berwarna format RGB dengan dimensi awal bervariasi yang kemudian diselaraskan ke matriks target piksel (224, 224, 3).

---

## Data Preprocessing
Sebelum citra dimasukkan ke dalam jaringan neural, serangkaian tahapan prapemrosesan dijalankan guna memastikan kestabilan optimasi gradien:
1. **Rescaling (Normalisasi Piksel)**: Nilai piksel yang semula berada pada rentang kecerahan integer [0-255] dikonversi menjadi rentang titik kambang (*floating-point*) [0.0 - 1.0] dengan mengalikan matriks citra lewat faktor skala `1./255`.
2. **Image Resizing**: Seluruh data gambar dipotong dan diubah ukurannya secara seragam menjadi dimensi $(224 \times 224)$ piksel agar kompatibel dengan dimensi input layer CNN.
3. **Data Splitting via Directory**: Memisahkan alur data citra ke dalam subset data latih (*Training Set*), data validasi (*Validation Set*), dan data uji (*Test Set*) untuk menjamin objektivitas evaluasi model.

---

## Modeling
Model dibangun menggunakan paradigma arsitektur *Sequential* TensorFlow yang disusun secara hierarkis sebagai berikut:

1. **Blok Konvolusi Pertama**: Lapisan `Conv2D` (32 filter, kernel $3 \times 3$, aktivasi `ReLU`) + `BatchNormalization()` + `MaxPool2D` ukuran $2 \times 2$.
2. **Blok Konvolusi Kedua**: Lapisan `Conv2D` (32 filter, kernel $4 \times 4$, aktivasi `ReLU`) + `BatchNormalization()` + `MaxPool2D` ukuran $2 \times 2$.
3. **Blok Konvolusi Ketiga**: Lapisan `Conv2D` (32 filter, kernel $7 \times 7$, aktivasi `ReLU`) + `BatchNormalization()` + `MaxPool2D` ukuran $2 \times 2$.
4. **Lapisan Perataan (Flatten)**: Mengubah tensor multi-dimensi menjadi vektor satu dimensi.
5. **Lapisan Terhubung Penuh (Fully Connected/Dense)**:
   * Lapisan tersembunyi 1: `Dense` (128 neuron, aktivasi `ReLU`) + `Dropout(0.5)`.
   * Lapisan tersembunyi 2: `Dense` (64 neuron, aktivasi `ReLU`) + `Dropout(0.3)`.
6. **Lapisan Output**: Lapisan `Dense` dengan 3 neuron menggunakan fungsi aktivasi `Softmax` untuk memetakan probabilitas prediksi distribusi multi-kelas.

Model dikompilasi menggunakan fungsi pengoptimal **RMSprop**, fungsi kerugian **Categorical Crossentropy**, serta metrik evaluasi **Accuracy**.

---

## Evaluasi & Hasil Analisis

### Proses Training
Melalui pemanfaatan *Custom Callback*, pelatihan dihentikan secara prematur pada **Epoch 9/30** karena kondisi ideal terpenuhi:
* **Training Accuracy**: 97.68%
* **Training Loss**: 0.0757
* **Validation Accuracy**: 96.09%
* **Validation Loss**: 0.1718

### Hasil Pengujian Independen (Test Dataset)
Berikut adalah rangkuman performa model yang diuji menggunakan data independen:

#### Classification Report:
```text
              precision    recall  f1-score   support

         Cat     0.9751    0.9786    0.9768      1119
         Dog     0.9859    0.9272    0.9556      1057
        Wild     0.9261    0.9781    0.9514      1050

    accuracy                         0.9616      3226
   macro avg     0.9623    0.9613    0.9613      3226
weighted avg     0.9627    0.9616    0.9616      3226
