from .lpca import LPCA
from .lego import LEGO
from .ldle import LDLE
from .guidotti import Guidotti
from .lpoly import LocalPoly
from .msvd import MultiscaleSVD
from .common import DEV, DT, to_t, knn_train, knn_query, MemTimer

METHODS = {"lpca": LPCA, "ldle": LDLE, "lego": LEGO, "guidotti": Guidotti, "lpoly": LocalPoly, "msvd": MultiscaleSVD}
