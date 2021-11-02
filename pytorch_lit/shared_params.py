from torch.nn import Parameter

from .memory import Memory


class SharedParameterUtil:
    _isHijacked = False
    _memory = None
    _mainNew = None

    @staticmethod
    def _shared_new(cls, data=None, requires_grad=True):
        if data is None:
            return SharedParameterUtil._mainNew(cls, data, requires_grad)
        mShape = data.shape
        fSize = 1
        for i in mShape:
            fSize *= i
        sharedSlot = Memory.obtain(SharedParameterUtil._memory)
        nT = sharedSlot.reshape(-1)[:fSize]
        nT = nT.reshape(mShape)
        return SharedParameterUtil._mainNew(cls, nT, requires_grad)

    @staticmethod
    def hijackParameters(memoryKey):
        if SharedParameterUtil._isHijacked:
            raise RuntimeError("already hijacked, reset first")
        SharedParameterUtil._mainNew = Parameter.__new__
        SharedParameterUtil._isHijacked = True
        SharedParameterUtil._memory = memoryKey
        Parameter.__new__ = SharedParameterUtil._shared_new

    @staticmethod
    def resetParameters(resetMemory=False):
        if not SharedParameterUtil._isHijacked:
            return
        Parameter.__new__ = SharedParameterUtil._mainNew
        SharedParameterUtil._isHijacked = False
        SharedParameterUtil._mainNew = None
        if resetMemory:
            Memory.deallocKey(SharedParameterUtil._memory)
        SharedParameterUtil._memory = None
