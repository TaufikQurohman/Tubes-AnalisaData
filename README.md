# 📊 Stepwise Linear Regression for Technician Salary Prediction

Tugas Besar Analisa Data (Kelompok 5) DS-47-01

Model regresi linear bertahap (Stepwise Regression) untuk memprediksi gaji teknisi berdasarkan beberapa variabel kemampuan akademik dan tahun kelulusan.

---

# 👥 Kelompok 5

- Muhammad Zikra Al Rizkya Adler — 103052300076
- Taufik Qurohman — 103052300055
- Hauzan Rafi Attallah — 103052330011

---

# 📖 Deskripsi Project

Project ini bertujuan untuk menganalisis hubungan antara:

- Tahun kelulusan
- Kemampuan Bahasa Inggris
- Logical Reasoning
- Computer Programming

terhadap:

- Gaji teknisi (Salary)

Metode yang digunakan adalah:

- Korelasi Pearson
- Regresi Linear
- Stepwise Regression

---

# 🚀 Features

- Data preprocessing
- Perhitungan korelasi manual
- Analisis hubungan antar variabel
- Stepwise Regression
- Perhitungan JKR & JKT
- Uji F
- Interpretasi model regresi
- Analisis R-Squared

---

# 📦 Requirements

Install dependencies terlebih dahulu:

```bash
pip install pandas numpy scipy openpyxl
```

---

# ▶️ Run Program

```bash
python main.py
```

atau jalankan notebook:

```bash
jupyter notebook
```

---

# 📁 Dataset Information

Dataset terdiri dari:

- 50 data teknisi
- 4 variabel independen
- 1 variabel dependen

## Variabel Bebas (Independent Variables)

| Variabel | Keterangan |
|---|---|
| X1 | GraduationYear |
| X2 | English |
| X3 | Logical |
| X4 | ComputerProgramming |

## Variabel Terikat (Dependent Variable)

| Variabel | Keterangan |
|---|---|
| Y | Salary |

---

# 🧹 Data Processing

## Import Library

```python
import pandas as pd
import numpy as np
import scipy.stats as st
import math
```

---

## Load Dataset

```python
data = pd.read_excel('Kel-5-Data-Gaji-teknisi.xlsx')
```

---

# 📊 Correlation Analysis

## Rumus Korelasi

```python
def calculate_correlation(x, y):
   n = len(x)

   sum_xi = sum(x)
   sum_xj = sum(y)

   sum_xi_squared = sum([xi**2 for xi in x])
   sum_xj_squared = sum([xj**2 for xj in y])

   sum_xi_xj = sum([x[i] * y[i] for i in range(n)])

   numerator = sum_xi_xj - (sum_xi * sum_xj) / n

   denominator_xi = sum_xi_squared - (sum_xi**2) / n
   denominator_xj = sum_xj_squared - (sum_xj**2) / n

   denominator = math.sqrt(denominator_xi * denominator_xj)

   if denominator == 0:
       return 0
   else:
       return numerator / denominator
```

---

## Correlation Table

| Variable | GraduationYear | English | Logical | ComputerProgramming | Salary |
|---|---|---|---|---|---|
| GraduationYear | 1.000 | -0.081 | 0.230 | -0.097 | -0.244 |
| English | -0.081 | 1.000 | 0.514 | 0.407 | 0.194 |
| Logical | 0.230 | 0.514 | 1.000 | 0.453 | 0.081 |
| ComputerProgramming | -0.097 | 0.407 | 0.453 | 1.000 | 0.111 |
| Salary | -0.244 | 0.194 | 0.081 | 0.111 | 1.000 |

---

# 📌 Correlation Interpretation

Korelasi tertinggi terjadi antara:

- English dan Logical → r = 0.514

Menunjukkan hubungan positif yang cukup kuat antara kemampuan Bahasa Inggris dan kemampuan logika.

Namun, seluruh variabel memiliki korelasi rendah terhadap Salary.

---

# 📈 Stepwise Regression

## Rumus JKR Sederhana

```python
def Jumlah_Kuadrat_Regresi(col1, col2, n):
  x = np.sum(col1)
  y = np.sum(col2)

  x_bar = x / n
  y_bar = y / n

  xy = np.sum(col1 * col2)
  x2 = np.sum(col1 ** 2)

  b = (n * xy - x * y)/(n * x2 - x**2)

  a = y_bar - b * x_bar

  yi = []

  for i in range(n):
    yi.append(a + b * col1[i])

  hasil = np.sum((yi - y_bar)**2)

  return hasil
```

---

## Rumus JKR Berganda

```python
def Jumlah_Kuadrat_Regresi_Berganda(xs, y):
  xs = [np.array(x) for x in xs]

  X_mat = np.column_stack([np.ones(len(y))] + xs)

  beta = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ y)

  Y_pred = X_mat @ beta

  JKR = np.sum((Y_pred - np.mean(y))**2)

  return JKR, beta
```

---

# 📉 Hasil Uji Stepwise

## Variabel yang Signifikan

Hanya:

- X1 (GraduationYear)

yang masuk ke dalam model regresi pada taraf signifikansi 10%.

Variabel:

- English
- Logical
- ComputerProgramming

tidak signifikan secara statistik terhadap Salary.

---

# 📌 Persamaan Regresi

```text
Y = 61541586.2102 - 30431.0345 X1
```

---

# 📊 Interpretasi Model

- Konstanta = 61541586.2102
- Koefisien X1 = -30431.0345

Artinya:

Setiap kenaikan 1 tahun GraduationYear akan menurunkan prediksi Salary sebesar sekitar Rp30.431.

---

# 📈 R-Squared

```text
R² = 0.0595
```

Artinya:

Model hanya mampu menjelaskan sekitar:

- 5.95% variasi Salary

Sisanya:

- 94.05%

dipengaruhi oleh faktor lain di luar model.

---

# 📌 Kesimpulan

- Tidak ada hubungan kuat antara variabel akademik dengan Salary.
- GraduationYear menjadi satu-satunya variabel yang masuk model.
- Model regresi memiliki performa yang sangat lemah.
- Dataset membutuhkan variabel tambahan agar prediksi Salary lebih akurat.

---

# 🛠️ Built With

- Python
- Pandas
- NumPy
- SciPy
- OpenPyXL

---

# 🎯 Purpose

Project ini dibuat untuk:

- Pembelajaran regresi linear
- Implementasi Stepwise Regression
- Analisis korelasi
- Tugas besar analisa data
- Statistik dan machine learning dasar

---

# 👨‍💻 Authors

- Muhammad Zikra Al Rizkya Adler
- Taufik Qurohman
- Hauzan Rafi Attallah
