class Layer:
    def __init__(self):
        self.training = True
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads
