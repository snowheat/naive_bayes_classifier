class PengklasifikasiNaiveBayesTeks():
    pass

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
            self.pnb = PengklasifikasiNaiveBayesTeks()
        elif jenis_data_set.lower() == 'id3':
            self.pnb = PengklasifikasiNaiveBayesTradisional()
        else:
            self.pnb = PengklasifikasiNaiveBayesTeks()
        pass

    def set_model_dari_data_latih(self):
        pass

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