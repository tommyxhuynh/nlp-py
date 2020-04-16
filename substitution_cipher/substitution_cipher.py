# Substitution Cipher
#TODO: write some description of what this is and how it works
# 
import random
import string
import re
import textwrap
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#TRAINING_CORPUS_PATH = "./data/train.txt"
TRAINING_CORPUS_PATH = "/tmp/bible.txt"

def main():
    # preprocess corpus
    #corpus = preprocess_corpus(TRAINING_CORPUS_PATH)
    #with open("/tmp/processed_bible.txt", "w") as fout:
    #    fout.write(" ".join(corpus))
    corpus = []
    with open("./temp/processed_moby_dick.txt", "r") as fin:
        corpus = fin.read().split(' ')

    # get bigram count using sklearn
    #ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
    #counts = ngram_vectorizer.fit_transform(words)
    #print(len(ngram_vectorizer.get_feature_names()))
    ##print (ngram_vectorizer.get_feature_names())
    #matrix = counts.toarray().astype(int)
    #sum1 = 0
    #sum2 = 0
    #i = 0
    #print(ngram_vectorizer.get_feature_names()[26])
    #for array in matrix:
    #    #sum1 += array[0]
    #    #sum2 += array[27]
    #    if array[26] > 0:
    #        print(words[i])
    #        exit()
    #    i += 1
    #print(sum1, sum2)
    #exit()

    # build bigram word model from training corpus
    bigram_word_model = BigramWordModel(corpus)
    #print(bigram_word_model.get_word_probability("jesus"))
    #print(bigram_word_model.get_word_probability("god"))
    #print(bigram_word_model.get_word_probability("love"))
    #print(bigram_word_model.get_word_probability("moby"))
    #print(bigram_word_model.get_word_probability("winste@d"))
    #exit(0)

    # build bigram sentence model
    bigram_sentence_model = BigramSentenceModel(corpus, bigram_word_model)
    #s1 = "I thought I would sail about a little and see the watery part of the world"
    #print(bigram_sentence_model.get_log_sentence_probability(s1))
    #s2 = "I little sail thought watery would the I about a and part see of the world"
    #print(bigram_sentence_model.get_log_sentence_probability(s2))
    #exit(0)

    s = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''
    s1 = " ".join(preprocess_str(s))
    # encrypt the message
    cipher = create_substitution_cipher()
    encrypted_text = encrypt(cipher, " ".join(preprocess_str(s1)))
    print(encrypted_text)
    run_genetic_algorithm(bigram_sentence_model, encrypted_text)
    exit(0)

    # search for the cipher map using genetic algorithm

    # decrypt the message
    decrypted_text = decrypt(cipher, encrypted_text)
    print(decrypted_text)

def preprocess_corpus(corpus_path):
    """Prepare corpus for training by stripping non-ASCII characters and converts to lowercase

    Parameters
    ----------
    corpus_path
        The path to the training corpus

    Returns
    -------
    str
        The preprocessed corpus containing only lowercase ascii alphabets
    """
    preprocessed_words = []
    for line in open(corpus_path):
        preprocessed_word.extend(preprocess_str(line))

def preprocess_str(line):
    preprocessed_words = []
    text = line.lower()
    text = text.translate(text.maketrans("", "", string.punctuation + string.octdigits))
    text = text.translate(text.maketrans("-—","  "))
    for word in text.split(' '):
        w = strip_non_ascii_alpha(word)
        if w == '':
            continue
        else:
            preprocessed_words.append(w)
    return preprocessed_words

class BigramWordModel:
    """
    A representation of the Bigram Word Model
    ...
    Attributes
    ----------
    __unigram_model
        The np array of character's probability
    __leading_unigram_model (BETA)
        The np array of character's probability being a leading character in a word
    __bigram_model
        The np 2D array mapping a bigram to its probability, where m[x,y] = p(y|x)
        the index is ord(char) - ord('a')

    Methods
    -------
    __init__(corpus_path)
        Create and train the model on the corpus given by corpus_path
    """
    __unigram_model = np.ones(26)
    __leading_unigram_model = np.ones(26)
    __bigram_model = np.ones((26, 26))

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus
            The training corpus as a list of words (str)
        """
        word_count = 0 # used to calculate leading unigram probability

        for word in corpus:
            # count leading unigram
            self.__leading_unigram_model[ord(word[0]) - ord('a')] += 1
            word_count += 1
            # count bigrams
            for i in range(0, len(word)-1):
                char1 = ord(word[i]) - ord('a')
                char2 = ord(word[i+1]) - ord('a')
                self.__unigram_model[char1] += 1
                self.__bigram_model[char1, char2] += 1

        # calculate leading unigram probability model
        self.__leading_unigram_model /= word_count
        # calculate unigram probability model
        self.__unigram_model /= self.__unigram_model.sum()
        # calculate bigram probability model
        self.__bigram_model /= self.__bigram_model.sum(axis=1, keepdims=True)

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

        #probability = self.__leading_unigram_model[ord(word[0]) - ord('a')]
        probability = self.__unigram_model[ord(word[0]) - ord('a')]
        for i in range(0, len(word)-1):
            char1 = ord(word[i]) - ord('a')
            char2 = ord(word[i+1]) - ord('a')
            probability *= self.__bigram_model[char1, char2]

        return probability

class BigramSentenceModel:
    """
    A representation of the Bigram Sentence Model
    ...
    Attributes
    ----------
    __unigram_model
        The dict mapping a word to its probability in the training corpus
    __bigram_model
        The dict mapping a bigram to its probability, where dict["a"]["b"] = p(b|a)

    Methods
    -------
    @TODO: write comments
    """
    __unigram_model = None
    __bigram_model = {}
    __corpus_size = 0

    def __init__(self, corpus, word_model):
        self.__unigram_model = word_model
        self.__corpus_size = len(corpus)
        for i in range(1, len(corpus)):
            word = corpus[i]
            # count bigram
            prev_word = corpus[i-1]
            if prev_word not in self.__bigram_model:
                self.__bigram_model[prev_word] = {}
            if word not in self.__bigram_model[prev_word]:
                self.__bigram_model[prev_word][word] = 2 # since we +1 smoothing
            else:
                self.__bigram_model[prev_word][word] += 1

    def get_log_sentence_probability(self, sentence_str):
        sentence = preprocess_str(sentence_str)
        ## use only word model
        probability = 0
        for word in sentence:
            probability += np.log(self.__unigram_model.get_word_probability(word))
        ## account for sentence model
        ## one may expect this to do better since it account for sentence structure, but it got stuck in local optimum
        ## more often
        #probability = np.log(self.__unigram_model.get_word_probability(sentence[0]))
        #for i in range(1, len(sentence)):
        #    prev_word = sentence[i-1]
        #    word = sentence[i-1]
        #    bigram_probability = np.log(1.0 / float(self.__corpus_size-1))
        #    if prev_word in self.__bigram_model and word in self.__bigram_model[prev_word]:
        #        bigram_probability = np.log(float(self.__bigram_model[prev_word][word]) / float(self.__corpus_size-1))
        #    probability += bigram_probability
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
        #elif letter == '-' or letter == '—'
        #    ascii_alphas.append(' ')
    return "".join(ascii_alphas)

def create_substitution_cipher():
    """Creates a substitution cipher

    Returns
    -------
    dict
        The cipher mapping plain characters to their encoded counterpart
    """
    plain_characters = list(string.ascii_lowercase)
    encoded_characters = list(string.ascii_lowercase)
    random.shuffle(encoded_characters)
    cipher = {}
    for i in range(26):
        cipher[plain_characters[i]] = encoded_characters[i]
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
        char = plain_text[i]
        if ord(char) < ord('a') or ord(char) > ord('z')+1:
            encrypted_text.append(char)
        else:
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

def run_genetic_algorithm(model, encoded_message):
    ### run an evolutionary algorithm to decode the message
    
    regex = re.compile('[^a-zA-Z]')
    letters1 = list(string.ascii_lowercase)
    # this is our initialization point
    dna_pool = []
    for _ in range(20):
        dna = list(string.ascii_lowercase)
        random.shuffle(dna)
        dna_pool.append(dna)
    
    def evolve_offspring(dna_pool, n_children):
        # make n_children per offspring
        offspring = []
      
        for dna in dna_pool:
            for _ in range(n_children):
                copy = dna.copy()
                j = np.random.randint(len(copy))
                k = np.random.randint(len(copy))
      
                # switch
                tmp = copy[j]
                copy[j] = copy[k]
                copy[k] = tmp
                offspring.append(copy)
      
        return offspring + dna_pool
    
    num_iters = 1000
    scores = np.zeros(num_iters)
    best_dna = None
    best_map = None
    best_score = float('-inf')
    for i in range(num_iters):
        if i > 0:
            # get offspring from the current dna pool
            dna_pool = evolve_offspring(dna_pool, 3)
      
        # calculate score for each dna
        dna2score = {}
        for dna in dna_pool:
            # populate map
            current_map = {}
            for k, v in zip(letters1, dna):
                current_map[k] = v
      
            decoded_message = decrypt(current_map, encoded_message)
            score = model.get_log_sentence_probability(decoded_message)
      
            # store it
            # needs to be a string to be a dict key
            dna2score[''.join(dna)] = score
      
            # record the best so far
            if score > best_score:
                best_dna = dna
                best_map = current_map
                best_score = score
      
        # average score for this generation
        scores[i] = np.mean(list(dna2score.values()))
      
        # keep the best 5 dna
        # also turn them back into list of single chars
        sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
        dna_pool = [list(k) for k, v in sorted_dna[:5]]
      
        if i % 200 == 0:
            print("iter:", i, "score:", scores[i], "best so far:", best_score)
    
    # use best score
    decoded_message = decrypt(best_map, encoded_message)
    
    #print("LL of decoded message:", model.get_log_sentence_probability(decoded_message))
    #print("LL of true message:", model.get_log_sentence_probability(regex.sub(' ', original_message.lower())))
    
    
    # which letters are wrong?
    #for true, v in true_mapping.items():
    #  pred = best_map[v]
    #  if true != pred:
    #    print("true: %s, pred: %s" % (true, pred))
    
    # print the final decoded message
    print("Decoded message:\n", textwrap.fill(decoded_message))
    
    #print("\nTrue message:\n", original_message)

if __name__ == "__main__":
    main()
