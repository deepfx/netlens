class Objective:

    def __init__(self, name: str = None):
        self.name = name

    def objective_function(self, x):
        raise NotImplementedError()

    @property
    def cls_name(self):
        return self.__class__.__name__

    def __call__(self, x):
        return self.objective_function(x)

    def __repr__(self):
        return self.name or self.cls_name
