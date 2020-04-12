# Substitution Cipher
#TODO: write some description of what this is and how it works
# 
import string

TRAINING_CORPUS_PATH = "./data/train.txt"

def main():
    # build bigram letter model from training corpus
    bigram_letter_model = get_bigram_letter_model(TRAINING_CORPUS_PATH)

    # build bigram word model

    # encrypt the message
    text = "holla"
    cipher = create_rotation_cipher(3)
    encrypted_text = encrypt(cipher, text)
    print(encrypted_text)

    # search for the cipher map using genetic algorithm

    # decrypt the message
    decrypted_text = decrypt(cipher, encrypted_text)
    print(decrypted_text)

def get_bigram_letter_model(corpus_path):
    """Builds the bigram letter model from the given corpus

    Parameters
    ----------
    corpus_path
        The path to the training corpus

    Returns
    -------
    dict
        The bigram probablity model where dict["ab"] = p(a|b)

    """
    # initialize unigram letter count (used in denominator for bigram probability)
    unigram_count = {}
    for letter in range(ord('a'), ord('z')+1):
        unigram_count[chr(letter)] = 0

    # initialize bigram letter sequence count 
    bigram_seq_count = {}
    for letter1 in range(ord('a'), ord('z')+1):
        for letter2 in range(ord('a'), ord('z')+1):
            seq = chr(letter1) + chr(letter2)
            bigram_seq_count[seq] = 0

    # count bigram sequences from training corpus
    with open(corpus_path, "r") as fin:
        for line in fin.readlines():
            # strip punctuation
            str = line.translate(line.maketrans('', '', string.punctuation))
            # strip digits
            str = str.translate(str.maketrans('', '', string.octdigits))
            str = str.lower()
            for word in str.split():
                for i in range(0, len(word)-1):
                    if not (is_ascii_alpha(word[i]) and is_ascii_alpha(word[i+1])):
                        # only account for alpha characters in bigram seq count
                        continue
                    unigram_count[word[i]] += 1
                    bigram_seq_count[word[i:i+2]] += 1

    # calculate bigram probability model
    bigram_probability_model = {}
    for bigram in bigram_seq_count:
        # applying +1 smoothing
        numerator = float(bigram_seq_count[bigram] + 1)
        denominator = float(unigram_count[bigram[0]] + 26)
        bigram_probability_model[bigram] = numerator / denominator

    return bigram_probability_model

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
