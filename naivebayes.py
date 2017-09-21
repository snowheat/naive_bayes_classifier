import pandas as pd
import pickle

class PengklasifikasiNaiveBayesTeks():

    def __init__(self, estimasi_parameter='mle'):
        self.estimasi_parameter=estimasi_parameter

    def set_model_dari_data_latih(self, data_latih):
        self.model = {}

        df_data_latih = pd.DataFrame(data_latih)

        self.jumlah_baris_data_latih = len(df_data_latih)

        self.kelas_unik = self._get_kelas_unik(df_data_latih)

        self.kata_unik = self._get_kata_unik(df_data_latih)

        self.jumlah_kata_unik = len(self.kata_unik)

        self.total_kata_unik = len(self.kata_unik)

        self.larik_prob_atribut_given_kelas = self._get_larik_prob_atribut_given_kelas()

        self.model = {'kelas':self.kelas_unik,'prob_atribut_given_kelas':self.larik_prob_atribut_given_kelas}

        #print(self.kelas_unik)
        print(self.kata_unik)
        #print(self.larik_prob_atribut_given_kelas)

        print(self.model)

    def _get_kata_unik(self, df_data_latih):
        kata_unik = {}

        for index,row in df_data_latih.iterrows():

            pecah_kata_per_baris = row[0].split()
            kelas_per_baris = row[1]

            for kata in pecah_kata_per_baris:
                if kata in kata_unik:
                    kata_unik[kata]['total'] += 1
                else:
                    kata_unik[kata] = {'total':1}
                    for kelas in self.kelas_unik:
                        kata_unik[kata][kelas] = 0

                kata_unik[kata][kelas_per_baris] += 1

        return kata_unik

    def _get_kelas_unik(self, df_data_latih_kelas):
        kelas_unik = {}

        for index, row in df_data_latih_kelas.iterrows():
            kelas_per_baris = row[1]
            if kelas_per_baris in kelas_unik:
                kelas_unik[kelas_per_baris]['total'] += 1
            else:
                kelas_unik[kelas_per_baris] = {'total':1,'total_kata':0,'prob':0}

            kelas_unik[kelas_per_baris]['total_kata'] += len(row[0].split())
            kelas_unik[kelas_per_baris]['prob'] = kelas_unik[kelas_per_baris]['total'] / self.jumlah_baris_data_latih

        return kelas_unik

    def _get_larik_prob_atribut_given_kelas(self):
        prob_atribut_given_kelas = {}

        for kata in self.kata_unik:
            if kata not in prob_atribut_given_kelas:
                prob_atribut_given_kelas[kata] = {}

            for kelas in self.kelas_unik:
                prob_atribut_given_kelas[kata][kelas] = self._get_likehood(kata, kelas)

        return prob_atribut_given_kelas

    def _get_likehood(self, kata, kelas):

        if self.estimasi_parameter is 'map':
            likelihood = self.get_estimasi_parameter_map(kata,kelas)


        if self.estimasi_parameter is 'mle':
            likelihood = self.get_estimasi_parameter_mle(kata,kelas)


        return likelihood

    def simpan_model_data_latih_ke_file_pickle(self,nama_file_pickle):
        with open(nama_file_pickle,'wb') as handle:
            pickle.dump(self.model,handle,protocol=pickle.HIGHEST_PROTOCOL)

        with open(nama_file_pickle, 'rb') as handle:
            b = pickle.load(handle)
            print("model dari pickle ",b)

    def get_estimasi_parameter_map(self,kata,kelas):
        likelihood = round((self.kata_unik[kata][kelas]+1) / (self.kelas_unik[kelas]['total_kata']+self.total_kata_unik), 3)
        return likelihood

    def get_estimasi_parameter_mle(self,kata,kelas):
        likelihood = round((self.kata_unik[kata][kelas]) / (self.kelas_unik[kelas]['total_kata']), 3)
        return likelihood


class PengklasifikasiNaiveBayesTradisional():
    pass


class PengklasifikasiNaiveBayes:
    """

    Oleh :
    Muhammad Insan Al-Amin ( )
    Sigit Kariagil ()

    * * * MENGGUNAKAN PYTHON > 3 * * *

    Class PengklasifikasiNaiveBayes menyediakan interface untuk pemodelan, penglasifikasian,
    dan pengetesan Naive Bayes. Method utama dalam class ini adalah :

        1. buat_model(atribut,data_pembelajaran): Membuat model Naive Bayes baru
        2. klasifikasi(data_untuk_diklasifikasi): Mengklasifikasikan data menggunakan model Naive Bayes yang telah dibuat
        3. tes(data_tes): Melakukan tes performa dari model Naive Bayes yang telah dibuat

    Args:
        jenis_data_set (str): Jenis data set yang digunakan ['tradisional','teks'].
        estimasi_parameter (str): Jenis estimasi parameter yang digunakan ['mle','map']

    Attributes:
        pnb : Instansiasi PengklasifikasiNaiveBayesTradisional() / PengklasifikasiNaiveBayesTeks()
        model : Wadah hasil pemodelan Pohon Keputusan

    """

    pnb = None
    model = None

    def __init__(self, jenis_data_set='teks',estimasi_parameter='mle'):

        print("* * * * * * * * * LOG PENGKLASIFIKASI NAIVE BAYES * * * * * * * * * ")
        print("* * Hasil Model & Prediksi ada setelah log ini * * ")

        if jenis_data_set.lower() == 'teks':
            self.pnb = PengklasifikasiNaiveBayesTeks(estimasi_parameter)
        elif jenis_data_set.lower() == 'tradisional':
            self.pnb = PengklasifikasiNaiveBayesTradisional(estimasi_parameter)
        else:
            self.pnb = PengklasifikasiNaiveBayesTeks()
        pass

    def set_model_dari_data_latih(self,data_latih):
        self.pnb.set_model_dari_data_latih(data_latih)

    def simpan_model_data_latih_ke_file_pickle(self, nama_file_pickle):
        self.pnb.simpan_model_data_latih_ke_file_pickle(nama_file_pickle)

    def set_model_dari_file(self):
        pass

    def get_likelihood(self):
        pass

    def get_prior(self):
        pass

    def simpan_model_ke_file(self):
        pass


    def klasifikasi(self):
        pass

    def tes(self):
        pass