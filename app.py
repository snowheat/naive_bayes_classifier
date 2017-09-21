from naivebayes import PengklasifikasiNaiveBayes







pnb = PengklasifikasiNaiveBayes(jenis_data_set='teks',estimasi_parameter='mle')

data_latih = [["chinese beijing chinese","yes"],
    ["chinese chinese shanghai","yes"],
    ["chinese macao","yes"],
    ["tokyo japan chinese","no"],]

pnb.set_model_dari_data_latih(data_latih)

pnb.simpan_model_data_latih_ke_file_pickle("contoh_model_chinese.pickle")




pnb = PengklasifikasiNaiveBayes(jenis_data_set='teks',estimasi_parameter='map')

pnb.set_model_dari_data_latih(data_latih)