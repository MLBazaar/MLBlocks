import importlib


class Function(object):

    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = kwargs

    def produce(self, X, **predict_params):
        package, name = self.function.rsplit('.', 1)
        function = getattr(importlib.import_module(package), name)

        kwargs = self.kwargs.copy()
        kwargs.update(predict_params)

        return function(X, **kwargs)
