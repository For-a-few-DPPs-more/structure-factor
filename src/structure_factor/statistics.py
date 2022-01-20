from structure_factor.structure_factor import StructureFactor


class SummeryStatistics:
    def __int__(self, list_point_pattern):
        self.list_point_pattern = list_point_pattern  # list of PointPattern
        self.s = len(list_point_pattern)  # number of sample

    def sample_mean(self, estimator, **params):
        return k, m

    def sample_variance(self, estimator, **params):
        # ivar = sum of var(k)
        return k, var, ivar

    def sample_bias(self, estimator, **params):
        return k, bias, ibas

    def sample_MSE(self, estimator, **params):
        k, var, ivar = self.sample_variance(self, estimator, **params)
        _, bias, ibias = self.sample_bias(self, estimator, **params)
        mse = var + bias ** 2
        imse = ivar + ibias ** 2
        return k, mse, imse
