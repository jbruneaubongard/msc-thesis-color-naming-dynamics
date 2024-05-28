import pickle
from zipfile import ZipFile
from .tools import *

LOGGER = get_logger('ib_naming_model')
DEFAULT_MODEL_URL = 'https://www.dropbox.com/s/70w953orv27kz1o/IB_color_naming_model.zip?dl=1'


def load_model(filename=None, model_dir='./models/'):
    ensure_dir(model_dir)
    if filename is None:
        filename = model_dir + 'IB_color_naming_model/model.pkl'
    if not os.path.isfile(filename):
        LOGGER.info('downloading default model from %s  ...' % DEFAULT_MODEL_URL)
        urlretrieve(DEFAULT_MODEL_URL, model_dir + 'temp.zip')
        LOGGER.info('extracting model files ...')
        with ZipFile(model_dir + 'temp.zip', 'r') as zf:
            zf.extractall(model_dir)
            os.remove(model_dir + 'temp.zip')
            os.rename(model_dir + 'IB_color_naming_model/IB_color_naming.pkl', filename)
    with open(filename, 'rb') as f:
        LOGGER.info('loading model from file: %s' % filename)
        model_data = pickle.load(f)
        return IBNamingModel(**model_data)


class IBNamingModel(object):

    def __init__(self, pM, pU_M, betas, IB_curve, qW_M):
        self.pM = pM if len(pM.shape) == 2 else pM[:, None]
        self.pU_M = pU_M
        self.betas = betas
        self.IB_curve = IB_curve
        self.qW_M = qW_M
        
    def complexity(self, pW_M):
        """
        :param pW_M: encoder (naming system)
        :return: I(M;W)
        """
        return MI(pW_M * self.pM)

    def accuracy(self, pW_M):
        """
        :param pW_M: encoder (naming system)
        :return: I(W;U)
        """
        pMW = pW_M * self.pM
        pWU = pMW.T @ self.pU_M
        return MI(pWU)
