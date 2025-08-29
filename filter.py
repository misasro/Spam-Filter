from collections import Counter
from typing import Iterable, Self

from pca import PCA
import utils
from corpus import Corpus
import preprocess as p
import numpy as np

STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
             'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
             'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
             'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
             'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
             'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
             "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
             'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
             "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
             "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
             "wouldn't"}


def cosine_sim(a: np.array, b: np.array) -> float:
    """
    Calculate similarity between two vectors using cosine of angle between them
    https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    :param a: first vector
    :param b: second vector
    :return: cosine of angle
    """

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class MyFilter:
    def __init__(self):
        self.vocabulary = set()
        self.word_to_index = {}
        self.truth_mail_cl = {}
        self.idf = np.empty
        self.pca = PCA(70)
        self.X = np.empty(0)
        self.y = np.empty(0)

    def preprocess(self, corpus_dir: str) -> tuple[list[str], list[list[str]]]:
        """
        Preprocess corpus to tuple of filenames and lists of words in emails
        :param corpus_dir: name of corpus directory
        :return: filenames, words
        """
        corpus = Corpus(corpus_dir)
        filenames, bodies = zip(*corpus.emails())
        bodies = map(p.extract_email_body, bodies)
        mail_words = map(p.extract_words, bodies)
        mail_words = list(mail_words)
        return filenames, mail_words

    def make_vocab(self, emails: Iterable[set[str]]) -> None:
        """
        Create vocabulary of all words of all emails in corpus
        :param emails: words in emails
        """
        for words in emails:
            self.vocabulary.update(words - STOPWORDS)
        self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}

    def term_freq(self, mail_words: Iterable[str]) -> np.array:
        """
        Create vector of term frequencies of words in email
        :param mail_words: words in email
        :return: vector of term frequencies
        """
        vector = np.zeros(len(self.vocabulary), dtype=int)
        # count number of occurrences of words from mail which are in the vocabulary
        word_count = Counter(filter(lambda x: x in self.vocabulary, mail_words))
        for word, count in word_count.items():
            vector[self.word_to_index[word]] = count
        return vector / len(vector)

    def prepare_idf(self, emails: list[set[str]]) -> None:
        """
        Calculate inverse document frequency of corpus
        :param emails: emails from training set
        """
        n_docs = len(emails)
        self.idf = np.zeros(len(self.word_to_index))
        for term, idx in self.word_to_index.items():
            num_of_docs_contains_term = sum(1 for doc in emails if term in doc)
            self.idf[idx] = np.log((n_docs + 1) / (num_of_docs_contains_term + 1)) + 1

    def tf_idf(self, emails: Iterable[list[str]]) -> np.array:
        """
        Encode mails to vectors
        :param emails: list of emails
        :return: array of encoded emails
        """
        return np.array([self.term_freq(x) * self.idf for x in emails])

    def undersample(self, emails: list[list[str]], y: np.array) -> (list[list[str]], np.array):
        """
        Balance spams and hams in training set
        :param emails: list of emails
        :param y: results
        :return: subset of emails and results
        """
        spam_indices = np.where(y == 1)[0]
        ham_indices = np.where(y == 0)[0]
        np.random.seed(42)
        n_spam = len(spam_indices)
        n_ham = len(ham_indices)

        # choose which indices to keep
        if n_spam > n_ham:
            np.random.shuffle(spam_indices)
            spam_indices = spam_indices[:n_ham]
        elif n_spam < n_ham:
            np.random.shuffle(ham_indices)
            ham_indices = ham_indices[:n_spam]

        # use the indices to reduce number of oversampled class (spam/ham)
        balanced_indices = np.concatenate([ham_indices, spam_indices])
        np.random.shuffle(balanced_indices)
        return [emails[i] for i in balanced_indices], y[balanced_indices]

    def train(self, train_corpus_dir: str) -> Self:
        """
        Save training set, make vocabulary and calculate idf for later prediction
        :param train_corpus_dir: training set
        """
        truth_fpath = train_corpus_dir + '/' + '!truth.txt'
        self.truth_mail_cl = utils.read_classification_from_file(truth_fpath)
        filenames, emails = self.preprocess(train_corpus_dir)
        self.y = np.array([int(self.truth_mail_cl[filename] == 'SPAM') for filename in filenames])
        emails, self.y = self.undersample(emails, self.y)
        doc_sets = [set(doc) for doc in emails]
        self.make_vocab(doc_sets)
        self.prepare_idf(doc_sets)
        mail_vectors = self.tf_idf(emails)
        self.X = np.vstack(mail_vectors)
        self.pca.fit(self.X)
        self.X = self.pca.transform(self.X)
        return self

    def predict_email(self, mail_vec: np.array, k=3) -> str:
        """ Predict if email is spam or ham according to nearest neighbours from training set
        :param mail_vec: vector
        :param k: number of nearest neighbours
        :return: string with prediction SPAM/OK
        """
        nearest = sorted(zip(self.X, self.y), reverse=True, key=lambda x: cosine_sim(x[0], mail_vec))[:k]
        num_of_spams = sum(is_spam for _, is_spam in nearest)
        if num_of_spams >= (k - num_of_spams):
            return 'SPAM'
        return 'OK'

    def predict(self, pred_corpus_dir: str) -> dict[str, str]:
        """
        Predict all emails in corpus
        :param pred_corpus_dir: name of set to predict
        :return: dictionary with filenames and predictions
        """
        filenames, words = self.preprocess(pred_corpus_dir)
        mails_vectors = self.tf_idf(words)
        mails_vectors = self.pca.transform(mails_vectors)
        predictions = {
            filenames[i]: self.predict_email(mails_vectors[i], 5)
            for i in range(len(filenames))
        }
        return predictions

    def test(self, test_corpus_dir) -> None:
        """
        Create file with predictions in corpus
        :param test_corpus_dir: name of corpus
        """
        predictions = self.predict(test_corpus_dir)
        utils.write_classification_to_file(test_corpus_dir + '/!prediction.txt', predictions)
