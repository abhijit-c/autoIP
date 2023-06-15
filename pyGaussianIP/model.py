from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def adjoint(self, x, *args, **kwargs):
        pass
