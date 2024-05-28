import string
import pandas as pd
from .tools import *
import os

LOGGER = get_logger('figures')

N_CHIPS = 330
N_COLS = 41
N_ROWS = 10
SPACE = 0.1

WCS_CIELAB_FILE = 'cnum-vhcm-lab-new.txt'
WCS_CNUM_FILE = 'chip.txt'
ROWS = [string.ascii_uppercase[i] for i in range(10)]


# init module
curr_path = os.path.dirname(os.path.abspath(__file__))
data_dir = curr_path + '/../data/'
ensure_dir(data_dir)
ensure_file(data_dir + WCS_CIELAB_FILE, 'http://www1.icsi.berkeley.edu/wcs/data/cnum-maps/cnum-vhcm-lab-new.txt')
ensure_file(data_dir + WCS_CNUM_FILE, 'http://www1.icsi.berkeley.edu/wcs/data/20021219/txt/chip.txt')

__CHIPS = pd.read_csv(data_dir + WCS_CIELAB_FILE, delimiter='\t').sort_values(by='#cnum')
WCS_CNUMS = pd.read_csv(data_dir + WCS_CNUM_FILE, delimiter='\t', header=None).values
WCS_CHIPS = __CHIPS[['L*', 'a*', 'b*']].values
WCS_CHIPS_RGB = lab2rgb(WCS_CHIPS)

CNUMS_WCS_COR = dict(zip(WCS_CNUMS[:, 0], [(ROWS.index(WCS_CNUMS[cnum - 1, 1]), WCS_CNUMS[cnum - 1, 2]) for cnum in WCS_CNUMS[:, 0]]))
_WCS_COR_CNUMS = dict(zip(WCS_CNUMS[:, 3], WCS_CNUMS[:, 0]))


def cnum2ind(cnum):
    """
    convert chip number to location in the WCS palette
    Example: cnum2ind(100) returns (2,22)
    WARNING: cnum is indexed from 1
    """
    return CNUMS_WCS_COR[cnum]

DF = pd.DataFrame({'cnum': np.arange(1,331)})

DF['x'] = DF['cnum'].apply(lambda x: cnum2ind(x)[0])
DF['y'] = DF['cnum'].apply(lambda x: cnum2ind(x)[1])
DF['R'] = DF['cnum'].apply(lambda x: WCS_CHIPS_RGB[x-1][0])
DF['G'] = DF['cnum'].apply(lambda x: WCS_CHIPS_RGB[x-1][1])
DF['B'] = DF['cnum'].apply(lambda x: WCS_CHIPS_RGB[x-1][2])

DF.set_index(['x', 'y'], inplace=True)


def ind2cnum(x, y):
    """
    convert location in the WCS palette to chip number
    Example: ind2cnum(2,22) returns 100
    """
    return DF.loc[x, y]['cnum']


def code2cnum(code):
    """
    convert WCS palette code to chip number
    Example: code2cnum('C22') returns 100
    :param c: string
    :return:
    """
    if code[0] == 'A':
        return _WCS_COR_CNUMS['A0']
    if code[0] == 'J':
        return _WCS_COR_CNUMS['J0']
    return _WCS_COR_CNUMS[code]


def grid2img(grid, small_grid=False, white_space=True):
    d = grid.shape[1]
    img = np.ones((N_ROWS, N_COLS + 1, d))
    if not white_space:
        img = img * np.nan
    for cnum in range(1, len(grid) + 1):
        i, j = cnum2ind(cnum)
        j = j + 1 if j > 0 else j
        img[i, j, :] = grid[cnum - 1, :]
        if img[i, j, 0] == 50:
            img[i, j, :] = img[i, j, :]
    if small_grid:
        img = img[1:-1, 2:]
    return img


def get_color_grid(pCW, chips=WCS_CHIPS):
    pC = pCW.sum(axis=1)[:, None]
    pW_C = np.where(pC > 0, pCW / pC, 1 / pCW.shape[1])
    y = pW_C.argmax(axis=1)
    pW = pCW.sum(axis=0)[:, None]
    pC_W = pCW.T / (pW + 1e-20)
    mu_w = lab2rgb(pC_W.dot(chips))
    grid = mu_w[y]
    grid[pC[:, 0] == 0] = np.nan * grid[pC[:, 0] == 0]
    return grid


# For Voronoi partitions

def clean_partition(partition):
    # Makes sure that the possible values are 0 to voc_size-1
    map = {word:i for i, word in enumerate(np.unique(partition))}
    return np.vectorize(map.get)(partition)


def partition_to_encoder(partition):
    # Converts the partition to a one-hot encoder

    partition = clean_partition(partition)

    voc_size = len(np.unique(partition))
    encoder = np.zeros((330, voc_size))

    for i in range(8):
        for j in range(41):
            if partition[i, j] == -1:
                print(f'Partition not complete at {i}, {j}')
            else:
                chip_index = int(ind2cnum(i+1, j) - 1)
                word_index = partition[i, j]
                encoder[chip_index, word_index] = 1.0
    # Add values for the 'outlier' chips (indices (0,0) and (9,0), corresponding to cnum 141 and 89)
    encoder[140, partition[0, 0]] = 1.0 # assign same value as (1,0)
    encoder[88, partition[7, 0]] = 1.0 # assign same value as (8,0)

    assert np.all(np.sum(encoder, axis=1) == 1.0), 'The probability distribution is not normalized!'

    return encoder


def encoder_to_partition(encoder):
    grid = np.full((10,41), np.nan)

    for cnum, word_distribution in enumerate(encoder):
        i, j = cnum2ind(cnum + 1)
        word = np.argmax(word_distribution)
        grid[i,j] = word
    
    return grid 
