import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    for index in range(test_set.num_items):
        best_score =float("-inf")
        best_guess =""
        prob_dict = {}

        X, lengths =test_set.get_item_Xlengths(index)
        for trained_word, trained_model in models.items():
            try:
                LogL_P = trained_model.score(X,lengths)
                prob_dict[trained_word]=logL_P
            except Exception as e:
                prob_dict[trained_word]=float("-inf")

            if LogL_P > best_score:
                best_score = LogL_P
                best_guess = trained_word

        probabilities.append(prob_dict)
        guesses.append(best_guess)
    return probabilities, guesses
    



    
