import email
import re

SUFFIXES = ["ization", "ational", "fulness", "iveness", "ousness",
            "ing", "est", "ity", "ion", "ive", "ous", "ize",
            "ed", "er", "ly", "es", "al",
            "s", "y"]


def extract_email_body(email_content: str) -> str:
    """
    Extract email body from email format using python email library
    :param email_content: whole email structure
    :return: string of email body
    """
    b = email.message_from_string(email_content)
    body = ""

    if b.is_multipart():
        # iterate over parts of the email
        for part in b.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            charset = part.get_content_charset()

            # if charset is invalid or None, default to utf-8
            if not charset or charset == "default_charset":
                charset = "utf-8"

            # check if the part is plain text and not an attachment
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body_bytes = part.get_payload(decode=True)
                if body_bytes:
                    try:
                        body = body_bytes.decode(charset, errors='replace')
                    except LookupError:
                        body = body_bytes.decode("utf-8",
                                                 errors='replace')
                break
    else:
        # single-part email
        body_bytes = b.get_payload(decode=True)
        charset = b.get_content_charset()

        # if charset is invalid or None, default to utf-8
        if not charset or charset == "default_charset":
            charset = "utf-8"

        if body_bytes:
            try:
                body = body_bytes.decode(charset, errors='replace')
            except LookupError:
                body = body_bytes.decode("utf-8", errors='replace')

    return body


def stem(word: str) -> str:
    """
    Remove suffixes like ing, ily
    :param word: word to be stemmed
    :return: word without suffix
    """
    for suff in SUFFIXES:
        if word.endswith(suff):
            return word[:len(suff)]
    return word


def extract_words(email_body: str) -> list[str]:
    """
    Extract words from email body using regex pattern with chars a-z and apostrophe
    :param email_body: string of email body
    :return: list of words in email
    """
    words = re.findall(r"[a-z']+", email_body.lower())
    words = map(stem, words)
    words = filter(lambda word: len(word) > 0, words)
    words = list(words)
    return words
