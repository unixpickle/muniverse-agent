from anyrl.models import CNN
from anyrl.models.util import impala_cnn


class IMPALAModel(CNN):
    def base(self, obs_batch):
        return impala_cnn(obs_batch)
