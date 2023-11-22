# Laporan Proyek Machine Learning - Faizah Mappanyompa

## Domain Proyek

![jamur](https://cdn.hourdetroit.com/wp-content/uploads/sites/20/2022/05/mushrooms.jpg)

<p align="justify">Jamur atau fungi merupakan jenis organisme yang dapat tumbuh di alam. Jamur dibagi menjadi dua jenis, yaitu beracun dan dapat dimakan. Di beberapa daerah pedesaan, jamur menjadi makanan pilihan bagi masyarakat yang hidup di sana. Kesalahan dalam memilih jamur dapat mengakibatkan potensi yang fatal seperti keracunan hingga kematian jika dikonsumsi. Sementara jamur yang aman dapat memberikan nutrisi bagi kesehatan tubuh. Menurut (TUTUNCU et.al, 2022) sulit bagi masyarakat untuk melakukan analisis biokimia setiap hari dan dikarenakan fitur morfologi yang dimiliki jamur mirip serta masyarakat pedesaan yang bukan ahli di bidangnya [1].

Pemanfaatan teknologi machine learning, prediksi jamur dapat mejadi solusi yang potensial untuk mengatasi masalah ini. Dengan pemilihan algoritma machine learning dan dataset yang tepat, dapat membuat suatu sistem yang dapat memprediksi apakah suatu jamur dapat dimakan atau tidak. Penelitian sebelumnya yang telah dilakukan oleh [(TUTUNCU et.al, 2022)][jurnal-1] dapat mengklasifikan jamur yang beracun dan dapat dimakan menggunakan model Decision Tree, Naive Bayes, AdaBoost, dan Support Vector Machine [1].

Bedasarkan latar belakang di atas, penulis mengembangkan model machine learning menggunakan model Random Forest untuk memprediksi jamur beracun dan tidak beracun yang dapat memberikan informasi handal kepada masyarakat.</p>

[jurnal-1]: https://www.researchgate.net/profile/Kemal-Tutuncu-2/publication/361490673_Edible_and_Poisonous_Mushrooms_Classification_by_Machine_Learning_Algorithms/links/62b98c456ec05339cca7d590/Edible-and-Poisonous-Mushrooms-Classification-by-Machine-Learning-Algorithms.pdf

## Business Understanding

<p align="justify">Pengembangan sistem prediksi jamur menggunakan algoritma machine learning Random Forest untuk membedakan jamur beracun dan dapat dimakan. Langkah ini merupakan langkah yang strategis untuk meningkatkan keamanan dalam mengumpulkan jamur yang dapat dikonsumsi. Solusi ini efektif bagi masyarakat pedesaan, penjelajah alam, hingga pemangku jual-beli yang tertarik dengan jamur.</p>

### Problem Statements

Bedasarkan latar belakang di atas, berikut beberapa rumusan masalah yang perlu diselesaikan:

- Bagaimana suatu sistem dapat memprediksi jamur yang beracun dan dapat dimakan?
- Bagaimana suatu sistem dapat meningkatkan keakuratan dan keandalan prediksi mengenai keberacunan jamur?

### Goals

Adapun tujuan dari penelitian ini sebagai berikut:

- Menghasilkan sistem yang dapat memprediksi jamur yang beracun dan dapat dimakan.
- Menghasilkan sistem yang memiliki tingkat keakuratan dan keandalan yang dapat memprediksi jamur beracun dan dapat dimakan.

### Solution Statement

- Menggunakan algoritma machine learning (Random Forest) dan deep learning (Neural Network) untuk melihat perbandingan akurasi kedua model.
- Mengukur performa model menggunakan metrik evaluasi seperti precision, recall, dan F-1.

## Data Understanding

Dataset yang digunakan berasal dari Kaggle [2]. Dataset ini berisi mengenai deskripsi dan sample dari 23 spesies jamur di family Agaricus dan Lepiota. Setiap spesies diidentifikasi sebagai poisonous (beracun) atau edible (dapat dimakan).
Adapun detail dataset sebagai berikut:

1. Memiliki jumlah 8124 data dengan 23 fitur.
2. Seluruh fitur bertipe data `object` atau kategorik.
3. Kolom target berupa `class` dengan nilai p (poisonus) dan e (edible).
4. Tugas yang dapat dilakukan: klasifikasi

### Variabel-variabel pada Mushroom Classification Kaggle dataset sebagai berikut:

Kelas Target:

1. Class : kolom yang menentukan apakah jamur beracun (p) atau dapat dimakan (e)

Kelas Feature:

1. cap-shape (bentuk topi jamur)

- bell:b (lonceng)
- conical:c (kerucut)
- convex:c (cembung)
- flat:f (datar)
- sunken:s (cekung)

2. cap-surface (bentuk permukaan topi jamur)

- fibrous=f (berserat)
- grooves=g (beralur)
- scaly=y (bersisik)
- smooth=s (halus)

3. cap-color (warna bentuk topi)

- brown=n (coklat)
- buff=b (kekuning-kuningan)
- cinnamon=c (coklat kayu manis)
- gray=g (abu-abu)
- green=r (hijau)
- pink=p (pink)
- purple=u (ungu)
- red=e (merah)
- white=w (putih)
- yellow=y (kuning)

4. bruises (memar)

- bruises=t (benar)
- no=f (tidak)

5. odor (bau)

- almond=a (almond)
- anise=l (manis)
- creosote=c (minyak kreosot)
- fishy=y (amis)
- foul=f (busuk)
- musty=m (apak)
- none=n (tidak bau)
- pungent=p (tajam/menusuk)
- spicy=s (pedas)

6. gill-attachment (ikatan bilah)

- attached=a (terikat)
- free=f (bebas)

7. gill-spacing (jarak bilah)

- close=c (dekat)
- crowded=w (ramai)

8. gill-size (ukuran bilah)

- broad=b (lebar)
- narrow=n (lurus)

9. gill-color (warna bilah)

- black=k (hitam)
- brown=n (cokelat muda)
- buff=b (kekuning-kuningan)
- chocolate=h (cokelat)
- gray=g (abu-abu)
- green=r (hijau)
- orange=o (oranye)
- pink=p (pink)
- purple=u (ungu)
- red=e (merah)
- white=w (putih)
- yellow=y (kuning)

10. stalk-shape (ukuran tangkai)

- enlarging=e (memperbesar)
- tapering=t (meruncing)

11. stalk-root (akar tangkai)

- bulbous=b (bulat)
- club=c (klub)
- cup=u (cangkir)
- equal=e (sama)
- rhizomorphs=z
- rooted=r (berakar)
- missing=? (tidak ada)

12. stalk-surface-above-ring (permukaan tangkai di atas cincin jamur)

- fibrous=f (berserta)
- scaly=y (bersisik)
- silky=k (halus)
- smooth=s (halus)

12. stalk-surface-below-ring (permukaan tangkai di bawah cincin jamur)

- fibrous=f (berserta)
- scaly=y (bersisik)
- silky=k (halus)
- smooth=s (halus)

13. stalk-color-above-ring (warna tangkai di atas cincin jamur)

- brown=n (coklat)
- buff=b (kekuning-kuningan)
- cinnamon=c (coklat kayu manis)
- gray=g (abu-abu)
- green=r (hijau)
- pink=p (pink)
- purple=u (ungu)
- red=e (merah)
- white=w (putih)
- yellow=y (kuning)

14. stalk-color-below-ring (warna tangkai di bawah cincin jamur)

- brown=n (coklat)
- buff=b (kekuning-kuningan)
- cinnamon=c (coklat kayu manis)
- gray=g (abu-abu)
- green=r (hijau)
- pink=p (pink)
- purple=u (ungu)
- red=e (merah)
- white=w (putih)
- yellow=y (kuning)

15. veil-type (tipe tudung jamur)

- partial=p (parsial)
- universal=u

16. veil-color (warna tudung jamur)

- brown=n (cokelat)
- orange=o (oranye)
- white=w (putih)
- yellow=y (kuning)

17. ring-number (jumlah cincin jamur)

- none=n (tidak ada)
- one=o (satu)
- two=t (dua)

18. ring-type (tipe cincin jamur)

- cobwebby=c (sarang laba-labar)
- evanescent=e (gelombang)
- flaring=f (terang)
- large=l (lebar)
- none=n (tidak ada)
- pendant=p (liontin)
- sheathing=s (pelapis)
- zone=z (berupa daerah)

19. spore-print-color (warna spora)

- black=k (hitam)
- brown=n (cokelat)
- buff=b (kekuning-kuningan)
- chocolate=h (cokelat)
- green=r (hijau)
- orange=o (oranye)
- purple=u (ungu)
- white=w (putih)
- yellow=y (kuning)

20. population (populasi)

- abundant=a (berlimpah)
- clustered=c (berkelompok)
- numerous=n (banyak)
- scattered=s (tersebar)
- several=v (beberapa)
- solitary=y (tersendiri)

21. habitat

- grasses=g (rumput-rumput)
- leaves=l (daun-daun)
- meadows=m (padang rumput)
- paths=p (jalanan)
- urban=u (perkotaan)
- waste=w (sampah)
- woods=d (hutan atau pepohonan)

  - Pada atribut diatas dilakukan EDA spesifiknya pada atribut target untuk mengetahui apakah data balance atau imbalance.

### EDA

#### Feature: class

Pada tahap ini, penulis melakukan analisis univariate terhadap satu feature yaitu `class` sebagai target. Periksa apakah kolom `class` yang merupakan atribut target itu imbalance atau balance. Kelas cukup balance karena diperoleh value p (poisonous) 3916 dan e (edible) 4208.

## Data Preparation

### Data Loading

1. Pada tahapan ini dilakukan import data Mushrooms Classification melalui Kaggle
2. Kemudian, import library yang dibutuhkan seperti pandas, seaborn, tensorflow, matplotlib, dan numpy.

### Feature Engineering

#### Label Encoder

Setelah eksplorasi data, seluruh kolom bersifat object dimana ini tidak dapat diproses model jika tidak diubah menjadi data numerical terlebih dahulu. Untuk mengubah data kategorik menjadi numerik digunakan `LabelEncoder` dari sklearn. `LabelEncoder` mengubah setiap unique values dalam kolom.

#### Train-Test Split

Lakukan pemisahan data train dan data test sebelum masuk ke tahap modelling. Pemisahan data dengan skala 80:20.

## Modeling

Penulis, memilih Random Forest dan Neural Network sebagai model dengan alasan berikut:

1. Random Forest merupakan algoritma supervised learning yang dapat mengklasifikasikan dan memprediksi dari beberapa pohon keputusan.
2. Neural Network merupakan algoritma yang dapat menangani data yang kompleks seperti non-linear. Dengan Neural Network memungkinkan untuk menyesuaikan arsitektur yang cocok dan sesuai untuk permasalahan identifikasi jamur.

### Random Forest (Machine Learning)

1. Menggunakan model Random Forest dengan `max_depth=3`. Untuk mencegah overfitting pada model, kedalaman keputusan random forest dibatasi sampai 3.
2. Train model menggunakan data train yang sudah displit
3. Lakukan prediksi terhadap data test `X_test`

### Neural Network (Deep Learning)

1. Neural network yang digunakan memiliki arsitektur sebagai berikut:
   - Memiliki layer dense sebesar 16 unit dengan activation function relu (Rectified Linear Unit). Menggunakan relu untuk non-linearitas.
   - Memiliki layer dense sebesar 16 unit dengan activation function relu.
   - Memiliki layer output sebesar 1 unit dengan activation function sigmoid, karena hasil output berupa apakah jamur beracun dan dapat dimakan (binary classification).
2. Konfigurasi model menggunakan optimizer `Adam` Algorithm dengan loss `binary_crossentropy` untuk mengklasifikasikan masalah biner, dalam hal ini output berupa apakah jamur poisonus atau edible.

## Evaluation

Untuk mengevaluasi performa model klasifikasi (confusion matrix), metrik evaluasi yang digunakan adalah:

### Accuracy

merepresentasikan model klasifikasi memberikan prediksi yang benar secara keseluruhan

![Accuracy](https://latex.codecogs.com/svg.latex?%5Cdpi%7B300%7D%20%5Csmall%20%5Cbg_white%20%5Cfn_cm%20%5Ctext%7BAccuracy%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%20%2B%20%5Ctext%7BTrue%20Negatives%7D%7D%7B%5Ctext%7BTotal%20Population%7D%7D)

### Precision

merepresentasikan prediksi positif model adalah benar, atau berapa persentase prediksi positif yang sebenarnya positif.
![Precision](https://latex.codecogs.com/svg.latex?%5Cdpi%7B300%7D%20%5Csmall%20%5Cbg_white%20%5Cfn_cm%20%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%7D%7B%5Ctext%7BTrue%20Positives%7D%20%2B%20%5Ctext%7BFalse%20Positives%7D%7D)

### Recall

merepresentasikan sejauh mana model dapat mengidentifikasi dengan benar semua instance yang seharusnya positif.

![Recall](https://latex.codecogs.com/svg.latex?%5Cdpi%7B300%7D%20%5Csmall%20%5Cbg_white%20%5Cfn_cm%20%5Ctext%7BRecall%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTrue%20Positives%7D%7D%7B%5Ctext%7BTrue%20Positives%7D%20%2B%20%5Ctext%7BFalse%20Negatives%7D%7D)

### F1 Score

gabungan dari recall dan presisi.

![F1 Score](https://latex.codecogs.com/svg.latex?%5Cdpi%7B300%7D%20%5Csmall%20%5Cbg_white%20%5Cfn_cm%20%5Ctext%7BF1%20Score%7D%20%3D%202%20%5Ctimes%20%5Cfrac%7B%5Ctext%7BPrecision%7D%20%5Ctimes%20%5Ctext%7BRecall%7D%7D%7B%5Ctext%7BPrecision%7D%20%2B%20%5Ctext%7BRecall%7D%7D)

#### Random Forest

<p align="center"><strong>Tabel 1</strong> <i>Classification Report Random Forest</i></p>

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Class 0      | 0.98      | 0.99   | 0.99     | 843     |
| Class 1      | 0.99      | 0.98   | 0.98     | 782     |
| Accuracy     |           |        | 0.98     | 1625    |
| Macro Avg    | 0.98      | 0.98   | 0.98     | 1625    |
| Weighted Avg | 0.98      | 0.98   | 0.98     | 1625    |

Berikut detail tabel <i>Classification Report</i> untuk <i>Random Forest</i>.

1. Class 0 (Edible atau jamur yang dapat dimakan)
   - Precision: 98%. Disebutkan sebelumnya, bahwa precision merupakan ukuran akurasi yang memprediksi data positif.
   - Recall: 99%
   - F1-Score: 99%
2. Class 1 (Poisonus atau jamur beracun)
   - Precision: 99%.
   - Recall: 98%
   - F1-Score: 98%
3. Akurasi keseluruhan model pada dataset: 98%
4. Rata-rata makro dan weight (bobot):
   - Precision, Recall, F-1 Score: 98% rata-rata pada seluruh kelas

#### Neural Network

<p align="center"><strong>Tabel 2</strong> <i>Classification Report Neural Network</i></p>

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Class 0      | 0.95      | 0.98   | 0.97     | 843     |
| Class 1      | 0.98      | 0.95   | 0.96     | 782     |
| Accuracy     |           |        | 0.96     | 1625    |
| Macro Avg    | 0.97      | 0.96   | 0.96     | 1625    |
| Weighted Avg | 0.96      | 0.96   | 0.96     | 1625    |

Berikut detail tabel <i>Classification Report</i> untuk <i>Neural Network</i>.

1. Class 0 (Edible atau jamur yang dapat dimakan)
   - Precision: 95%
   - Recall: 98%
   - F1-Score: 97%
2. Class 1 (Poisonus atau jamur beracun)
   - Precision: 98%.
   - Recall: 95%
   - F1-Score: 96%
3. Akurasi keseluruhan model pada dataset: 96%
4. Rata-rata makro dan weight (bobot):
   - Precision: 97% untuk rata-rata makro
   - Precision: 97% untuk rata-rata weight
   - Recall dan F-1 Score: 96% rata-rata pada seluruh kelas

Kedua model menghasilkan akurasi yang cukup tinggi. Namun, jika dilihat dari keseluruhan nilai `accuracy` maka Random Forest memiliki akurasi yang lebih tinggi sebesar 98% dibandingkan Neural Network sebesar 96%.

Bedasarkan problem statement diatas, kedua solusi sudah terpenuhi:

1. Suatu sistem dapat memprediksi jamur beracun dan dapat dimakan menggunakan algoritma machine learning seperti Random Forest atau deep learning seperti Neural Network. Model ini dilatih dan belajar pola dari dataset mengenai fitur-fitur yang dimiliki oleh jamur untuk membedakannya.
2. Sistem dapat meningkatkan akurasi dan keandalan dengan cara mengoptimalkan parameter model seperti hyperparameter tuning, dan pemilihan optimizer. Validasi melalui metrik evaluasi juga digunakan untuk mengukur kualitas prediksi.

## Conclusion

Dalam mengatasi permasalahan prediksi jamur beracun dan dapat dimakan, dua solusi telah diajukan: menggunakan model Random Forest dan Neural Network.

Random Forest dapat memberikan prediksi yang stabil dengan memanfaatkan sejumlah besar pohon keputusan. Penggunaan parameter, seperti max_depth=3, membantu mencegah overfitting dan meningkatkan interpretasi model.

Di sisi lain, Neural Network mampu menangani kompleksitas pola data yang tinggi dan dapat mengekstrak fitur secara otomatis. Dengan optimalisasi hyperparameter Neural Network dapat memberikan solusi yang adaptif dan efektif untuk kasus identifikasi jamur.

Solusi pemilihan model tergantung pada karakteristik data dan tujuan spesifik. Hasil akhir dari sebuah model harus dievaluasi menggunakan metrik accuracy, precision, recall, dan F1 score. Keseluruhan, penelitian ini bertujuan untuk memberikan panduan terkait keamanan jamur, meningkatkan kesadaran masyarakat, dan mengurangi risiko keracunan dalam memilih jamur.

## References

[1] K. Tutuncu, I. Cinar, R. Kursun, and M. Koklu, “Edible and poisonous mushrooms classification by machine learning algorithms,” 2022 11th Mediterranean Conference on Embedded Computing (MECO), 2022. doi:10.1109/meco55406.2022.9797212

[2] UCI Machine Learning Repository, "Mushroom Classification Dataset," Kaggle, 2023.
Available: https://www.kaggle.com/uciml/mushroom-classification
