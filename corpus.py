import os


class Corpus:

    def __init__(self, dir_name):
        self.dir = dir_name

    def emails(self):
        """
        Generator of emails.
        :return: name of email file, email file
        """
        for filename in os.listdir(self.dir):
            if filename.startswith('!'):
                continue
            filepath = os.path.join(self.dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                body = file.read()
                yield filename, body
