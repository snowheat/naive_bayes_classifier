import pandas as pd
import pickle


class PengklasifikasiNaiveBayesTeks():
    """
    Inisiasi parameter, default -> mle
    """
    def __init__(self, estimasi_parameter='mle'):
        self.estimasi_parameter = estimasi_parameter

    """
    Set model dari data latih
    :type
    data_latih          : list
    df_data_latih       : DataFrame
    kelas_unik          : dictionary
    kata_unik           : dictionary
    jumlah_kata_unik    : integer
    total_kata_unik     : integer
    larik_prob_atribut_given_kelas : dictionary
    model               : dictionary
    """
    def set_model_dari_data_latih(self, data_latih):
        self.model = {}

        df_data_latih = pd.DataFrame(data_latih)

        self.jumlah_baris_data_latih = len(df_data_latih)

        self.kelas_unik = self._get_kelas_unik(df_data_latih)

        self.kata_unik = self._get_kata_unik(df_data_latih)

        self.jumlah_kata_unik = len(self.kata_unik)

        self.total_kata_unik = len(self.kata_unik)

        self.larik_prob_atribut_given_kelas = self._get_larik_prob_atribut_given_kelas()

        self.model = {'kelas': self.kelas_unik, 'prob_atribut_given_kelas': self.larik_prob_atribut_given_kelas}

        # print(self.kelas_unik)
        # print(self.kata_unik)
        # print(self.larik_prob_atribut_given_kelas)

        print(self.model)

    """
    Mendapatkan kata-kata yang unik untuk setiap data latih
    """
    def _get_kata_unik(self, df_data_latih):
        kata_unik = {}

        # Hitung jumlah kata untuk tiap baris
        for index, row in df_data_latih.iterrows():

            pecah_kata_per_baris = row[0].split()
            kelas_per_baris = row[1]

            # Pisahkan kata, dan hitung jumlahnya
            for kata in pecah_kata_per_baris:
                if kata in kata_unik:
                    kata_unik[kata]['total'] += 1
                else:
                    kata_unik[kata] = {'total': 1}
                    for kelas in self.kelas_unik:
                        kata_unik[kata][kelas] = 0

                kata_unik[kata][kelas_per_baris] += 1

        return kata_unik

    """
    Mendapatkan kelas yang unik untuk setiap data latih
    """
    def _get_kelas_unik(self, df_data_latih_kelas):
        kelas_unik = {}

        # Perulangan untuk menghitung kelas unik
        # Jika sudah masuk dalam kelas_unik, increment nilai totalnya
        # Jika belum masuk kelas unik, tambahkan ke dalam kelas unik
        for index, row in df_data_latih_kelas.iterrows():
            kelas_per_baris = row[1]
            if kelas_per_baris in kelas_unik:
                kelas_unik[kelas_per_baris]['total'] += 1
            else:
                kelas_unik[kelas_per_baris] = {'total': 1, 'total_kata': 0, 'prob': 0}

            kelas_unik[kelas_per_baris]['total_kata'] += len(row[0].split())
            kelas_unik[kelas_per_baris]['prob'] = kelas_unik[kelas_per_baris]['total'] / self.jumlah_baris_data_latih

        return kelas_unik

    """
    Menghitung probability dari tiap larik 
    """
    def _get_larik_prob_atribut_given_kelas(self):
        prob_atribut_given_kelas = {}

        # Perulangan untuk menghitung probability kata unik
        # Jika kata belum masuk prob_atribut_given_kelas, masukkan kata tersebut
        # Hitung likelihood kelas untuk setiap kata unik
        for kata in self.kata_unik:
            if kata not in prob_atribut_given_kelas:
                prob_atribut_given_kelas[kata] = {}

            for kelas in self.kelas_unik:
                prob_atribut_given_kelas[kata][kelas] = self._get_likehood(kata, kelas)

        return prob_atribut_given_kelas

    """
    Menghitung likelihood  untuk tiap kata berdasarkan parameternya
    """
    def _get_likehood(self, kata, kelas):

        if self.estimasi_parameter is 'map':
            likelihood = self.get_estimasi_parameter_map(kata, kelas)

        if self.estimasi_parameter is 'mle':
            likelihood = self.get_estimasi_parameter_mle(kata, kelas)

        return likelihood

    """
    Menyimpan model ke file 
    """
    def simpan_model_ke_file(self, nama_file_pickle):
        with open(nama_file_pickle, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    Menghitung estimasi untuk map 
    p(w|c) = (jumlah kemunculan kata + 1)/ (jumlah total keseluruhan kata dalam kelas + jumlah total kata unik) 
    """
    def get_estimasi_parameter_map(self, kata, kelas):
        likelihood = round(
            (self.kata_unik[kata][kelas] + 1) / (self.kelas_unik[kelas]['total_kata'] + self.total_kata_unik), 3)
        return likelihood

    """
    Menghitung estimasi untuk mle
    p(w|c) = (jumlah kemunculan kata)/ (jumlah total keseluruhan kata dalam kelas) 
    """
    def get_estimasi_parameter_mle(self, kata, kelas):
        likelihood = round((self.kata_unik[kata][kelas]) / (self.kelas_unik[kelas]['total_kata']), 3)
        return likelihood

    """
    Set model dari file 
    """
    def set_model_dari_file(self, nama_file_pickle):
        with open(nama_file_pickle, 'rb') as handle:
            b = pickle.load(handle)
            print("model dari pickle ", b)

    """
    Mendapatkan nilai likelihood dari model 
    """
    def get_likelihood(self):
        return self.model['prob_atribut_given_kelas']


    """
    Mendapatkan nilai prior dari model 
    """
    def get_prior(self):
        prior = {}
        for k, v in self.model['kelas'].items():
            prior[k] = v['prob']
        return prior

    """
    Proses klasifikasi dari data uji 
    """
    def klasifikasi(self, data_uji):
        argmax = 0
        kelas_argmax = {}
        kelas_dipilih = None
        likelihood = self.get_likelihood()

        for kelas, nilai in self.get_prior().items():

            prob_posterior = nilai
            print(prob_posterior)

            for nilai_data_uji in data_uji.split():

                print("*", likelihood[nilai_data_uji][kelas])
                if likelihood[nilai_data_uji][kelas] > 0:
                    prob_posterior *= likelihood[nilai_data_uji][kelas]

            print(prob_posterior)

            prob_posterior = round(prob_posterior, 8)

            if prob_posterior > argmax:
                argmax = prob_posterior
                kelas_dipilih = kelas

            kelas_argmax[kelas] = prob_posterior

        print(kelas_argmax)

        return kelas_dipilih


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

    def __init__(self, jenis_data_set='teks', estimasi_parameter='mle'):

        print("* * * * * * * * * LOG PENGKLASIFIKASI NAIVE BAYES * * * * * * * * * ")
        print("* * Hasil Model & Prediksi ada setelah log ini * * ")

        if jenis_data_set.lower() == 'teks':
            self.pnb = PengklasifikasiNaiveBayesTeks(estimasi_parameter)
        elif jenis_data_set.lower() == 'tradisional':
            self.pnb = PengklasifikasiNaiveBayesTradisional(estimasi_parameter)
        else:
            self.pnb = PengklasifikasiNaiveBayesTeks()
        pass

    def set_model_dari_data_latih(self, data_latih):
        self.pnb.set_model_dari_data_latih(data_latih)

    def simpan_model_ke_file(self, nama_file_pickle):
        self.pnb.simpan_model_ke_file(nama_file_pickle)

    def set_model_dari_file(self, nama_file_pickle):
        self.pnb.set_model_dari_file(nama_file_pickle)

    def get_likelihood(self):
        return self.pnb.get_likelihood()

    def get_prior(self):
        return self.pnb.get_prior()

    def klasifikasi(self, data_uji):
        return self.pnb.klasifikasi(data_uji)
