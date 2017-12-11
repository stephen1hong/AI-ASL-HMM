import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    ---
    where -"L" is the likelihood of the fitted model, 
          -"p" is the number of parameters in model (i.e., model "complexity"),
          -"N" is the number of data points,
          -"p * log N" is the penalty term,
          -The term -2*LogL decreases with increasing model complexity (more parameters),
          -The penalty term p*logN increases with increasing model complexity
 
     Selection in BIC Model: 
          Lower the BIC score, the "better" the model. 
    
    How to calculate P?
   
     p (free parameters) =  n_states * (n_states -1) (the transition probabilities)
                 + n_states-1 (starting probabilities)
                 + n_states *n_features (Means)
                 + n_states *n_features (Variance)
                 = n_states * n_states + 2 * n_states * n_features -1
    where n_states denotes the model component within range(min_n_components, max_n_components);
        n_features denotes number of model variables. 

    References:
    [1] Project Q&A https://www.youtube.com/watch?v=EyTM0e2DlEM&feature=youtu.be
    [2] http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#
    [3] https://discussions.udacity.com/t/understanding-better-model-selection/232987/7
    [4] http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf 
    ---
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # idea: the lower the BIC score the better the model. 
        # SelectorBIC selects a model by accepting argument of ModelSelector instance of base class 
        # Loop from min_n_components to max_n_components 
        # Find the lowest BIC score as the better model. 

        lowest_bic =  float("inf") #np.infty
        best_model=None
        n_components_range = range(self.min_n_components, self.max_n_components + 1)

        for num_states in n_components_range:
            bic_scores=[]
            bic_model = None
            logL = None
            try:
                #bic_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    #random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                bic_model = self.base_model(num_states)
                logL = bic_model.score(self.X, self.lengths)
                N = self.X.shape[0]              # number of data points
                num_features = self.X.shape[1]   #num of variables
                p = num_states * num_states + 2 * num_states * num_features - 1
                bic_scores = -2*logL + p * math.log(N) 

                #look for lowest bic_score
                if bic_scores < lowest_bic: 
                    lowest_bic = bic_scores  
                    best_model = bic_model
            except Exception as e:
                break  
                
        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    -----
    From Eq. (17) at Reference [1], DIC is actually a difference between the likehood of the data (the first term) 
    and the average of anti-likehood of the data ( the second term) in the above formula. 
    
    The formula aboe is equivalent to: 
    DIC = log(P(original word)) - average(log(P(other words)))

    Selection using DIC Model: 
         The higher the DIC score, the better the model. 

    References:
    [1] Biem, Alain. "A model selection criterion for classification: application to HMM topology optimization"
        https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    [2] Project Q&A https://www.youtube.com/watch?v=EyTM0e2DlEM&feature=youtu.be
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # idea: the higher the DIC model, the better the model
        # SelectorDIC select a model by accepting argument of ModelSelector instance of base class 
        # Loop in range(min_n_components, max_n_components) 
        # Find the highest DIC score as the better model. 

        n_components_range = range(self.min_n_components, self.max_n_components + 1)
        M=len(n_components_range)
        try: 
            best_score =float("-inf") 
            best_model =None            
            for num_states in n_components_range:
                model = self.base_model(num_states) #compute a hmm model using base model
                LogL_P = model.score(self.X, self.lengths) #logP(X(i)))
                scores =[]
                for word in self.words:
                    if word !=self.this_word:
                        scores.append(model.score(self.X, self.lengths)) #SUM(log(P(X(all but i))
                
                # compute: DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))   
                dic_score = LogL_P -np.mean(scores)
                
                if dic_score > best_score:
                    best_score =dic_score
                    best_model =model

        except Exception as e:
            pass

        if best_model is None: 
            return self.base_model(self.n_constant)
        else:
            return best_model

    
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    References:
    [0] Project Q&A  https://www.youtube.com/watch?v=EyTM0e2DlEM&feature=youtu.be
    [1] http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-inf") 
        best_model = None
        n_splits = 3 #default 
        n_components_range = range(self.min_n_components, self.max_n_components + 1)

        for num_states in n_components_range:
            cv_scores = []
            logL = None

            # Check Data Length
            if len(self.sequences) < n_splits:
                break
            KF = KFold(n_splits=n_splits, random_state=self.random_state)
            try:
                for train_idx, test_idx in KF.split(self.sequences):

                    x_train, size_train = combine_sequences(train_idx, self.sequences)
                    x_test, size_test = combine_sequences(test_idx, self.sequences)

                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state,verbose=False).fit(x_train, size_train)
                    logL = hmm_model.score(x_test, size_test)
                
                cv_scores.append(logL)
            except Exception as e:
                pass

            if len(cv_scores) > 0:
                avg = np.average(cv_scores)
            else:
                avg = float("-inf")
                    
            if avg > best_score:
                best_model = hmm_model
                best_score = avg

        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model
