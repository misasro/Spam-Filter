def read_classification_from_file(filepath):
    """
    Read classification from file in directory.
    :param filepath: path to file
    :return: dictionary of mails with their classification
    """
    mail_classification = {}
    with open(filepath, 'r', encoding='UTF-8') as file:
        for line in file:
            mail, cl = line.split()
            mail_classification[mail] = cl
    return mail_classification


def write_classification_to_file(filepath, mail_classification: dict):
    """
    Write classification to file in directory.
    :param filepath: path to file
    :param mail_classification: dictionary of mails with their classification
    """
    res = ''
    for k, v in mail_classification.items():
        res += k + ' ' + v + '\n'
    with open(filepath, 'w', encoding='UTF-8') as file:
        file.write(res)
