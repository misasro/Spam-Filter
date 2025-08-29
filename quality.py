from utils import read_classification_from_file
from confmat import BinaryConfusionMatrix
import os


def quality_score(tp: int, tn: int, fp: int, fn: int):
    """
    Calculate quality score from values of confusion matrix.
    :param tp: num of true positives
    :param tn: num of true negatives
    :param fp: num of false positives
    :param fn: num of false negatives
    :return: score with high emphasis on false positives
    """
    return (tp + tn) / (tp + tn + 10 * fp + fn)


def compute_quality_for_corpus(corpus_dir):
    """
    Compute quality according to files ``!truth.txt``, ``!prediction.txt`` from corpus.
    :param corpus_dir: name of corpus directory
    :return: quality score of corpus
    """
    truth_mail_cl = {}
    pred_mail_cl = {}
    for filename in os.listdir(corpus_dir):
        file_path = corpus_dir + '/' + filename
        if filename.startswith('!truth.txt'):
            truth_mail_cl = read_classification_from_file(file_path)
        if filename.startswith('!prediction.txt'):
            pred_mail_cl = read_classification_from_file(file_path)

    conf_matrix = BinaryConfusionMatrix(pos_tag='SPAM', neg_tag='OK')
    conf_matrix.compute_from_dicts(truth_mail_cl, pred_mail_cl)
    return quality_score(conf_matrix.tp, conf_matrix.tn, conf_matrix.fp, conf_matrix.fn)
