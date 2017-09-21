from naivebayes import PengklasifikasiNaiveBayes




file_model = "contoh_model_chinese.pickle"


pnb = PengklasifikasiNaiveBayes(jenis_data_set='teks',estimasi_parameter='mle')

data_latih = [["chinese beijing chinese","yes"],
    ["chinese chinese shanghai","yes"],
    ["chinese macao","yes"],
    ["tokyo japan chinese","no"],]

data_uji = ["chinese chinese chinese tokyo japan"]

"""
Membuat model dari data latih
"""
pnb.set_model_dari_data_latih(data_latih)

"""
Menyimpan model dari data latih ke file
"""
pnb.simpan_model_ke_file(file_model)

"""
Menggunakan model dari file
"""
pnb.set_model_dari_file(file_model)

"""
Menampilkan nilai likelihood hasil training
"""
print(pnb.get_prior())

"""
Menampilkan nilai prior hasil training
"""
print(pnb.get_likelihood())


"""
Hasil klasifikasi data uji
"""
print(pnb.klasifikasi(data_uji))