
import numpy
import warnings

from scipy import spatial, optimize

class DetectionError:
    """
    Class implementing a common detection error metric. This is done by solving
    a linear sum assignment problem using the Hungarian algorithm. Each detections
    in the `truth` and `predicted` coordinates are associated in a one-vs-one manner.

    The `DetectionError` implements various metric calculation given the
    association
    """
    def __init__(self, truth, predicted, algorithm='hungarian', **kwargs):
        """
        Instantiates the `CentroidDetectionError` object
        :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
        :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
        :param algorithm: A `string` specifying the assignation algorithm to use, can be
                          either 'nearest' or 'hungarian'
        """
        # Assign member variables
        self.truth = truth
        self.predicted = predicted
        self.algorithm = algorithm
        self.default_scores = [
            "true_positive",
            "false_positive",
            "false_negative",
            "fnr",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "jaccard"
        ]

        # Assign kwargs variables as member variables
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Verifies if attribute exists, else defaults in hungarian
        try:
            self.assign = getattr(self, f"_assign_{algorithm}")
        except AttributeError:
            warnings.warn(f"The chosen algorithm `{algorithm}` does not exist. Defaults in `hungarian`.", category=UserWarning)
            self.assign = getattr(self, "_assign_hungarian")

        # Computes the cost matrix
        self.compute_cost_matrix()

    def _assign_nearest(self, threshold):
        """
        Assigns each truth detections to its nearest predicted detections. We consider
        a positive detections if it lies within a certain distance threshold.

        :param threshold: A `float` of threshold to apply

        :returns : A `tuple` of truth and predicted coupled
        """
        raveled_cost_matrix = self.cost_matrix.ravel()
        dist_sorted = numpy.argsort(raveled_cost_matrix)
        truth_couple, pred_couple = [], []
        for arg in dist_sorted:
            if raveled_cost_matrix[arg] > threshold:
                break
            where = (arg // self.cost_matrix.shape[1], arg - arg // self.cost_matrix.shape[1] * self.cost_matrix.shape[1])
            if (where[0] in truth_couple) or (where[1] in pred_couple):
                continue
            truth_couple.append(where[0])
            pred_couple.append(where[1])
        return truth_couple, pred_couple

    def _assign_hungarian(self, threshold, maximize=False):
        """
        Assigns each truth detections to its nearest predicted detections. We consider
        a positive detections if it lies within a certain distance threshold. The
        assignement uses the hungarian algorithm.

        See `scipy.optimize.linear_sum_assignment` for more details about hungarian algo.

        :param threshold: A `float` of threshold to apply
        :param maximize: (optional) Whether to maximize the assignement
        :returns : A `tuple` of truth and predicted coupled
        """
        truth_indices = numpy.arange(self.cost_matrix.shape[0])
        pred_indices = numpy.arange(self.cost_matrix.shape[1])

        # We remove all points without neighbors in a radius of value `threshold`
        if maximize:
            false_positives = numpy.sum(self.cost_matrix >= threshold, axis=0) == 0
            false_negatives = numpy.sum(self.cost_matrix >= threshold, axis=1) == 0
        else:
            false_positives = numpy.sum(self.cost_matrix < threshold, axis=0) == 0
            false_negatives = numpy.sum(self.cost_matrix < threshold, axis=1) == 0

        # Remove all false positives and false negatives
        cost = self.cost_matrix[~false_negatives][:, ~false_positives]
        truth_indices = truth_indices[~false_negatives]
        pred_indices = pred_indices[~false_positives]

        # Apply the hungarian algorithm,
        # using log on the distance helps getting better matches
        # Because of the log, we need to ensure there is no Distance of 0
        if maximize:
            truth_couple, pred_couple = optimize.linear_sum_assignment(cost, maximize=maximize)
        else:
            truth_couple, pred_couple = optimize.linear_sum_assignment(numpy.log(cost + 1e-6), maximize=maximize)

            # Check if all distances are smaller than the threshold
            distances = cost[truth_couple, pred_couple]
            truth_couple = truth_couple[distances < threshold]
            pred_couple = pred_couple[distances < threshold]

        truth_couple = truth_indices[truth_couple]
        pred_couple = pred_indices[pred_couple]

        return truth_couple, pred_couple

    def compute_cost_matrix(self):
        """
        Not implemented.
        Needs to be implemented in super class.
        """
        raise NotImplementedError("Implement in supered class!")

    def get_coupled(self):
        """
        Retreives the coupled indices of the truth and predicted
        :returns : A `tuple` of truth and predicted coupled
        """
        return self.truth_couple, self.pred_couple

    def get_false_positives(self):
        """
        Retreives the indices of the false positive detections
        :returns : A `list` of indices that are false positive detections
        """
        if self.cost_matrix.shape[1] > 0:
            return numpy.array(list(set(range(self.cost_matrix.shape[1])) - set(self.pred_couple)))
        return numpy.array([])

    def get_false_negatives(self):
        """
        Retreives the indices of the false negative detections
        :returns : A `list` of indices that are false negative detections
        """
        if self.cost_matrix.shape[0] > 0:
            return numpy.array(list(set(range(self.cost_matrix.shape[0])) - set(self.truth_couple)))
        return numpy.array([])

    def get_score_summary(self, scores=None):
        """
        Computes all the scores in a `dict`
        :param scores: A `list` of scores to return
        :returns : A `dict` of scores
        """
        if not scores:
            scores = self.default_scores
        summary = {score : getattr(self, score) for score in scores}
        return summary

    @property
    def true_positive(self):
        """
        Computes the number of true positive
        :returns : The number of true positive
        """
        return len(self.pred_couple)

    @property
    def false_positive(self):
        """
        Computes the number of false_positive
        :returns : The number of false positive
        """
        return self.cost_matrix.shape[1] - self.true_positive

    @property
    def false_negative(self):
        """
        Computes the number of false negative
        :returns : The number of false negative
        """
        return self.cost_matrix.shape[0] - self.true_positive

    @property
    def tpr(self):
        """
        Computes the true positive rate between the truth and the predictions
        :returns : A true positive rate score
        """
        return self.recall

    @property
    def fnr(self):
        """
        Computes the false negative rate between the truth and the predictions
        :returns : A false negative rate score
        """
        return 1 - self.recall

    @property
    def fpr(self):
        """
        Computes the false positive rate between the truth and the predictions
        NOTE. In the case of truth detections there are no true negatives
        :returns : A false positive rate
        """
        warnings.warn("Using the false positive rate as a score metric in the case truth predictions does not make sense as there are no true negative labels.",
                        category=UserWarning)
        return self.false_positive / self.false_negative

    @property
    def accuracy(self):
        """
        Computes the accuracy between the truth and the predictions
        :returns : An accuracy score
        """
        # No truth and no prediction is an accuracy of 1
        if all([shape == 0 for shape in self.cost_matrix.shape]):
            return 1.
        # Add numerical stability with 1e-6
        return len(self.pred_couple) / (self.cost_matrix.shape[0] + 1e-6)

    @property
    def precision(self):
        """
        Computes the precision between the truth and the predictions.
        :returns : A precision score
        """
        if all([shape == 0 for shape in self.cost_matrix.shape]):
            # Same behavior as in default sklearn
            return 0.
        # Add numerical stability with 1e-6
        return self.true_positive / (self.true_positive + self.false_positive + 1e-6)

    @property
    def recall(self):
        """
        Computes the recall between the truth and the predictions.
        :returns : A recall score
        """
        if all([shape == 0 for shape in self.cost_matrix.shape]):
            # Same behavior as in default sklearn
            return 0.
        # Add numerical stability with 1e-6
        return self.true_positive / (self.true_positive + self.false_negative + 1e-6)

    @property
    def f1_score(self):
        """
        Computes the F1-score between the truth and the predictions.
        :returns : A F1-score
        """
        prec = self.precision
        rec = self.recall
        return 2 * (prec * rec) / (prec + rec + 1e-6)

    @property
    def dice(self):
        """
        Computes the dice coefficient between the truth and the predictions.
        :returns : A dice coefficient score
        """
        return self.f1_score

    @property
    def jaccard(self):
        """
        Computes the Jaccard Index between the truth and the predictions.
        :returns : A Jaccard index score
        """
        f1 = self.f1_score
        return f1 / (2 - f1)

    @property
    def average_precision(self):
        """
        Computes the average precision between the truth and the predictions.

        :returns : An average precision score
        """
        # Add numerical stability with 1e-6
        return self.true_positive / (self.true_positive + self.false_negative + self.false_positive + 1e-6)


class CentroidDetectionError(DetectionError):
    """
    Class implementing common detection scores based on the centroid of the
    detected objects.
    """
    def __init__(self, truth, predicted, threshold, algorithm='nearest'):
        """
        Instantiates the `CentroidDetectionError` object
        :param truth: A 2D `numpy.ndarray` of coordinates of the true positions
        :param predicted: A 2D `numpy.ndarray` of coordinates of the predicted positions
        :param threshold: A distance threshold to consider a true positive detection (pixels)
        :param algorithm: A `string` specifying the assignation algorithm to use, can be
                          either 'nearest' or 'hungarian'
        """
        super().__init__(
            truth=truth,
            predicted=predicted,
            algorithm=algorithm
        )

        self.truth_couple, self.pred_couple = self.assign(threshold=threshold)

    def compute_cost_matrix(self):
        """
        Computes the cost matrix between all objects
        """
        # Returns truth_couple and pred_couple to 0 if truth or predicted are empty
        if (len(self.truth) < 1) or (len(self.predicted) < 1):
            self.cost_matrix = numpy.ones((len(self.truth), len(self.predicted))) * 1e+6
        else:
            self.cost_matrix = spatial.distance.cdist(self.truth, self.predicted, metric='euclidean')
