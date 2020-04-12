# Substitution Cipher
#TODO: write some description of what this is and how it works
# 
import string

TRAINING_CORPUS_PATH = "./data/train.txt"

def main():
    # build bigram word model from training corpus
    bigram_word_model = BigramWordModel(TRAINING_CORPUS_PATH)
    print(bigram_word_model.get_word_probability("jesus"))
    print(bigram_word_model.get_word_probability("moby"))
    print(bigram_word_model.get_word_probability("winste@d"))

    # build bigram sentence model

    # encrypt the message
    text = "holla"
    cipher = create_rotation_cipher(3)
    encrypted_text = encrypt(cipher, text)
    print(encrypted_text)

    # search for the cipher map using genetic algorithm

    # decrypt the message
    decrypted_text = decrypt(cipher, encrypted_text)
    print(decrypted_text)

class BigramWordModel:
    """
    A representation of the Bigram Word Model
    ...
    Attributes
    ----------
    __prefix_unigram_model
        The dict mapping a character to its probability as a leading character in words
    __bigram_model
        The dict mapping a bigram to its probability, where dict["ab"] = p(a|b)

    Methods
    -------
    train(corpus_path)
        Train the model on the corpus given by corpus_path
    """
    __prefix_unigram_model = {}
    __bigram_model = {}

    def __init__(self, corpus_path):
        """
        Parameters
        ----------
        corpus_path
            The path to the training corpus
        """
        # initialize unigram letter count (used in denominator for bigram probability)
        # and prefix unigram count (used in calculating prefix unigram probability)
        unigram_count = {}
        prefix_unigram_count = {}
        for letter in range(ord('a'), ord('z')+1):
            unigram_count[chr(letter)] = 0
            prefix_unigram_count[chr(letter)] = 0

        # initialize bigram letter sequence count 
        bigram_seq_count = {}
        for letter1 in range(ord('a'), ord('z')+1):
            for letter2 in range(ord('a'), ord('z')+1):
                seq = chr(letter1) + chr(letter2)
                bigram_seq_count[seq] = 0

        word_count = 0 # used to calculate prefix unigram probability

        # count bigram sequences from training corpus
        with open(corpus_path, "r") as fin:
            for line in fin.readlines():
                # strip punctuation
                str = line.translate(line.maketrans('', '', string.punctuation))
                # strip digits
                str = str.translate(str.maketrans('', '', string.octdigits))
                str = str.lower()
                for word in str.split():
                    if is_ascii_alpha(word[0]):
                        # update prefix unigram model
                        prefix_unigram_count[word[0]] += 1
                        word_count += 1
                    # update bigram model
                    for i in range(0, len(word)-1):
                        if not (is_ascii_alpha(word[i]) and is_ascii_alpha(word[i+1])):
                            # only account for alpha characters in bigram seq count
                            continue
                        unigram_count[word[i]] += 1
                        bigram_seq_count[word[i:i+2]] += 1

        # calculate prefix unigram probability model
        for letter in range(ord('a'), ord('z')):
            # apply +1 smoothing
            self.__prefix_unigram_model[chr(letter)] = float(prefix_unigram_count[chr(letter)]+1) / float(word_count+26)

        # calculate bigram probability model
        for bigram in bigram_seq_count:
            # apply +1 smoothing
            numerator = float(bigram_seq_count[bigram] + 1)
            denominator = float(unigram_count[bigram[0]] + 26)
            self.__bigram_model[bigram] = numerator / denominator

    def get_word_probability(self, word):
        """Calculates the probability of a word using the given bigram word model

        Parameters
        ----------
        word
            The word to calculate probability for (in lowercase)

        Returns
        -------
        float
            The probability of the word
        """
        word = strip_non_ascii_alpha(word)
        if len(word) == 0:
            return 1 # don't penalize non-ascii-alpha string

        probability = self.__prefix_unigram_model[word[0]]
        for i in range(0, len(word)-1):
            probability *= self.__bigram_model[word[i:i+2]]
        return probability

def is_ascii_alpha(str):
    """Checks if all the characters in the given str are ASCII letters
    
    Parameters
    ----------
    str
        The string to check for ASCII letters

    Returns
    -------
    bool
        True if all the characters are ASCII characters; False otherwise
    """
    return all(ord(c) >= ord('a') and ord(c) <= ord('z') for c in str)

def strip_non_ascii_alpha(str):
    """Removes non-ASCII-alpha characters from the given string

    Parameters
    ----------
    str
        The string to remove non-ASCII-alpha characters from

    Returns
    -------
    str
        The string with just ASCII alpha characters
    """
    ascii_alphas = []
    for letter in str:
        if is_ascii_alpha(letter):
            ascii_alphas.append(letter)
    return "".join(ascii_alphas)

def create_rotation_cipher(offset):
    """Creates a substitution cipher based on a rotation offset

    Parameters
    ----------
    offset
        The offset by which to "rotate" the characters when encrypting

    Returns
    -------
    dict
        The cipher
    """
    cipher = {}
    for key in range(ord('a'), ord('z')):
        value = key + offset
        # wrap back to a once passed z
        if value > ord('z'):
            value = ord('a') + value - ord('z')
        cipher[chr(key)] = chr(value)
    return cipher

def encrypt(cipher, plain_text):
    """Encrypts the plain text given a cipher

    Parameters
    ----------
    cipher : dict
        The cipher to use for encryption
    plain_text : str
        The text to encrypt

    Returns
    -------
    str
        The encrypted text
    """
    encrypted_text = []
    for i in range(0, len(plain_text)):
        encrypted_text.append(cipher[plain_text[i]])
    return "".join(encrypted_text)

def decrypt(cipher, encrypted_text):
    """Decrypts the encrypted text into plain text

    Parameters
    ----------
    cipher : dict
        The cipher used to encrypt the text
    encrypted_text : str
        The encrypted text

    Returns
    -------
    str
        The plain text
    """
    reverse_cipher = {}
    for key in cipher:
        reverse_cipher[cipher[key]] = key
    return encrypt(reverse_cipher, encrypted_text)

if __name__ == "__main__":
    main()
