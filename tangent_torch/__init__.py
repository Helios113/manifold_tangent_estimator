from .lpca import LPCA
from .lego import LEGO
from .ldle import LDLE
from .signature import Signature
from .common import DEV, DT, to_t, knn_train, knn_query, MemTimer

METHODS = {"lpca": LPCA, "ldle": LDLE, "lego": LEGO, "sig": Signature}
