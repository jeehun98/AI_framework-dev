class Tensor:
    """
    심볼릭 토큰 클래스, 튜플 형태 데이터 구조?

    >>> Tensor(shape=(3,4), dtype="float32")
    >>> Tensor.shape
    (3,4)
    >>> Tensor.dtype
    float32
    """

    def __init__(
        self,
        shape,
        dtype="float32",
        name=None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    def reshape(self, newshape):
        pass

    def squeeze(self, axis=None):
        pass

    def __add__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __matmul__(self, other):
        pass

    def __rmatmul__(self, other):
        pass

    def __div__(self, other):
        pass

    def __rdiv__(self, other):
        pass

    def __neg__(self):
        pass

    def __abs__(self):
        pass

    def __pow__(self, other):
        pass

    def __rpow__(self, other):
        pass

    