"""
Ensemble Model for prediciton only
"""

class EnsembleModel:
    """
    Class that represents a ensemble of predictive model
    """
    def __int__(self, models):
        self.models = models

    def start_train(self):
        for model in self.models:
            model.train()

    def start_prediction(self):
        for model in self.models:
            model.eval()

    def validate_batch(self, data, target):
        """
        Method that validates a batch of data
        :param data:
        :param target:
        :return:
        """
        #for model in self.models:
        #    output, _ = model.predict_batch(data)
        #output = output.mean()

        #return output, pred, None

    def predict_batch(self, data):
        """
        Method that predicts a batch of data
        :param data:
        :return:
        """
        output_list = []
        for model in self.models:
            output = model(data)
            output_list.append(output)
        pred = output[0].data.max(1)[1]  # get the index of the max log-probability
        return output, pred