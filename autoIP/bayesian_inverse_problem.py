class BayesianIP:
    def __init__(self, model, prior, likelihood, data):
        self.model = model
        self.prior = prior
        self.likelihood = likelihood
        self.data = data
