from typing import Dict


class BinaryConfusionMatrix:
    """
    Class to save values of binary confusion matrix.
    """

    def __init__(self, pos_tag, neg_tag):
        self.pos_tag = pos_tag
        self.neg_tag = neg_tag
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def as_dict(self):
        """
        Make dictionary of values of binary confusion matrix.
        """
        return {'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn}

    def update(self, truth, prediction):
        """
        Update values of confusion matrix.
        :param truth: real labels
        :param prediction:  predicted labels
        """
        if truth != self.pos_tag and truth != self.neg_tag:
            raise ValueError
        if prediction != self.pos_tag and prediction != self.neg_tag:
            raise ValueError
        if truth == self.pos_tag:
            if prediction == self.pos_tag:
                self.tp += 1
            else:
                self.fn += 1
        elif truth == self.neg_tag:
            if prediction == self.pos_tag:
                self.fp += 1
            else:
                self.tn += 1

    def compute_from_dicts(self, truth_dict: Dict, pred_dict: Dict):
        """
        Compute values of confusion matrix for whole directory.
        :param truth_dict: dictionary of real labels
        :param pred_dict: dictionary of predicted labels
        """
        if truth_dict.keys() != pred_dict.keys():
            raise ValueError
        for key in truth_dict.keys():
            self.update(truth_dict[key], pred_dict[key])
