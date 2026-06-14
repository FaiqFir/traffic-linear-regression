# ================================================
# ANALISIS REGRESI LINEAR BERGANDA
# Judul : Faktor-Faktor yang Mempengaruhi
#         Tingkat Kemacetan Lalu Lintas
# Dataset: Traffic.csv
# ================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ------------------------------------------------
# FUNGSI UTILITAS TAMPILAN  (semua lebar = 65)
# ------------------------------------------------
L = 65

def garis():
    print('+' + '-' * L + '+')

def judul_panel(teks):
    pad_kiri  = (L - len(teks)) // 2
    pad_kanan = L - pad_kiri - len(teks)
    print('|' + ' ' * pad_kiri + teks + ' ' * pad_kanan + '|')

def isi_panel(teks):
    sisa = L - len(teks) - 1
    if sisa < 0:
        teks = teks[:L - 4] + '...'
        sisa = 0
    print('| ' + teks + ' ' * sisa + '|')

def panel(judul, isi_list):
    print()
    garis()
    judul_panel(judul)
    garis()
    for baris in isi_list:
        isi_panel(baris)
    garis()

def garis_tabel(col_lebar):
    print('+' + '+'.join('-' * (w + 2) for w in col_lebar) + '+')

def baris_tabel(cols, col_lebar):
    print('|' + '|'.join(' ' + str(c).center(w) + ' '
                          for c, w in zip(cols, col_lebar)) + '|')

def tabel(judul, headers, rows, col_lebar):
    total = sum(col_lebar) + len(col_lebar) * 3 + 1
    pad   = (total - len(judul)) // 2
    print()
    print(' ' * pad + judul)
    garis_tabel(col_lebar)
    baris_tabel(headers, col_lebar)
    garis_tabel(col_lebar)
    for row in rows:
        baris_tabel(row, col_lebar)
        garis_tabel(col_lebar)

# ------------------------------------------------
# 1. MUAT DATASET
# ------------------------------------------------
df = pd.read_csv('Traffic.csv')

peta = {'low': 1, 'normal': 2, 'high': 3, 'heavy': 4}
df['Tingkat_Kemacetan'] = df['Traffic Situation'].map(peta)

# ------------------------------------------------
# 2. SIAPKAN VARIABEL X DAN Y
# ------------------------------------------------
X = df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']]
Y = df['Tingkat_Kemacetan']
X = sm.add_constant(X)

# ------------------------------------------------
# 3. JALANKAN ANALISIS REGRESI
# ------------------------------------------------
model  = sm.OLS(Y, X).fit()
Y_pred = model.predict(X)
koef   = model.params
r2     = model.rsquared
fstat  = model.fvalue
fprob  = model.f_pvalue
valid  = "VALID" if fprob < 0.05 else "TIDAK VALID"

nama_indo = {
    'const'     : 'Konstanta',
    'CarCount'  : 'Jumlah Mobil',
    'BikeCount' : 'Jumlah Motor',
    'BusCount'  : 'Jumlah Bus',
    'TruckCount': 'Jumlah Truk'
}

# ------------------------------------------------
# 4. TAMPILAN TERMINAL
# ------------------------------------------------

# Header utama
print()
garis()
judul_panel('ANALISIS REGRESI LINEAR BERGANDA')
garis()
judul_panel('Faktor yang Mempengaruhi Tingkat Kemacetan')
garis()

# Info dataset
panel('INFO DATASET', [
    f'  Jumlah Data   : {len(df)} baris',
    f'  Jumlah Kolom  : {len(df.columns)} kolom',
    f'  Sumber        : Traffic.csv',
])

# 5 data pertama — col total = 10+10+9+11+9 + 5*3 + 1 = 65
tabel(
    '5 Data Pertama',
    ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Kemacetan'],
    [
        [row['CarCount'], row['BikeCount'], row['BusCount'],
         row['TruckCount'], row['Tingkat_Kemacetan']]
        for _, row in df[['CarCount', 'BikeCount', 'BusCount',
                           'TruckCount', 'Tingkat_Kemacetan']].head().iterrows()
    ],
    col_lebar=[10, 10, 9, 11, 9]
)

# Hasil regresi
panel('HASIL REGRESI', [
    f'  R-squared   : {r2:.4f} ({r2*100:.2f}%)',
    f'  Keterangan  : {r2*100:.2f}% kemacetan dijelaskan 4 variabel',
    f'  F-statistic : {fstat:.4f}',
    f'  Prob(F)     : {fprob:.4f}  ->  Model {valid}',
])

# Tabel koefisien — col total = 14+11+9+18 + 4*3 + 1 = 65
rows_koef = []
for var in ['const', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']:
    p   = model.pvalues[var]
    c   = koef[var]
    ket = 'Berpengaruh' if p < 0.05 else 'Tidak Berpengaruh'
    rows_koef.append([nama_indo[var], f'{c:.4f}', f'{p:.4f}', ket])

tabel(
    'Koefisien Tiap Variabel',
    ['Variabel', 'Koefisien', 'P-value', 'Keterangan'],
    rows_koef,
    col_lebar=[14, 11, 9, 18]
)

# Persamaan regresi
panel('PERSAMAAN REGRESI', [
    f'  Y = {koef["const"]:.4f}',
    f'    + {koef["CarCount"]:.4f}  x Jumlah Mobil',
    f'    + {koef["BikeCount"]:.4f}  x Jumlah Motor',
    f'    + {koef["BusCount"]:.4f}  x Jumlah Bus',
    f'    + {koef["TruckCount"]:.4f}  x Jumlah Truk',
    '',
    '  Ket Y: 1=Lancar  2=Normal  3=Padat  4=Macet Total',
])

# Kesimpulan
koef_x   = koef.drop('const').abs().sort_values(ascending=False)
nama_dom = nama_indo[koef_x.index[0]]
panel('KESIMPULAN', [
    f'  Variabel dominan : {nama_dom} (koef = {koef[koef_x.index[0]]:.4f})',
    f'  R-squared        : {r2*100:.2f}%  ->  Model cukup baik',
])

# ------------------------------------------------
# 5. SIMPAN GRAFIK
# ------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(Y, Y_pred, alpha=0.4, color='steelblue', label='Data')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()],
         'r--', linewidth=2, label='Garis Ideal')
plt.xlabel('Nilai Aktual Tingkat Kemacetan')
plt.ylabel('Nilai Prediksi Tingkat Kemacetan')
plt.title('Perbandingan Nilai Aktual vs Prediksi')
plt.legend()
plt.tight_layout()
plt.savefig('grafik_prediksi.png')

plt.figure(figsize=(7, 5))
sns.heatmap(
    df[['CarCount', 'BikeCount', 'BusCount',
        'TruckCount', 'Tingkat_Kemacetan']].corr(),
    annot=True, cmap='Blues', fmt='.2f'
)
plt.title('Korelasi Antar Variabel')
plt.tight_layout()
plt.savefig('grafik_korelasi.png')

panel('OUTPUT FILE', [
    '  grafik_prediksi.png  ->  tersimpan',
    '  grafik_korelasi.png  ->  tersimpan',
])
print()