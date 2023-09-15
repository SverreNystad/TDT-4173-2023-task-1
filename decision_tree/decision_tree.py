import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class Inference:
    """
    Inference is a class that can be used to make predictions
    
    Example usage:
    >>> r1 = Rule([('Outlook', 'Overcast')], 'Yes')
    >>> r2 = Rule([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No')
    """

    def __init__(self, antecedent: list[tuple[str, str]], consequent: str):
        """
        antecedent: a list of tuples of the form (attr, val)
        The antecedent is the set of conditions that must be met for the consequent to be true
        consequent: a string label
        """
        self.antecedent = antecedent
        self.consequent = consequent

    def __repr__(self):
        return f"{' ^ '.join([f'{attr}={val}' for attr, val in self.antecedent])} => {self.consequent}"
    

class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        # 1. Calculate the entropy of every attribute a of the data set S.
        

        for attribute in range(len(X.columns)):
            feature = X.columns[attribute]
            print(feature)

            print(gain(X, feature, y))
        # 2. Partition ("split") the set S into subsets using the attribute for which the resulting entropy after splitting is minimized; or, equivalently, information gain is maximum.
        # 3. Make a decision tree node containing that attribute.
        # 4. Recurse on subsets using the remaining attributes

        # Repeatedly select the best features to split the data
        # Stop splitting, eh. when reaching:
        # 1. max depth
        # 2. min number of samples for each split
        # 3. min number of samples for each leaf
        # 4. Maximum number of features
        # Pruining could be applied to control complexity
        # Inference
        # 1. Navigate to a leaf node using the splitting 

        # TODO: Implement 
        # raise NotImplementedError()
        pass
    
    def predict(self, X: pd.DataFrame):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        raise NotImplementedError()
    
    def get_rules(self) -> list[Inference]:
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()

def gain(S: pd.DataFrame, A: str, y: pd.DataFrame, plus="Yes", minus="No") -> float:
    """
    Computes the information gain of a partitioning with respect to an attribute

    Args:
        S (DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
        A (str): attribute
        y (DataFrame): a vector of discrete ground-truth labels
        plus (str): the label for the positive class
        minus (str): the label for the negative class
    Returns:
        A positive float scalar corresponding to the information gain
        of the partitioning.
    """
    # The discrete values of the samples with respect to the attribute A
    values = np.unique(S[A])
    total_count = len(S)

    outcomes = np.unique(y)
    s_tot = np.zeros(len(outcomes))
    sub_gains = []
    for value in values:
        plus_count = 0
        minus_count = 0
        for row in range(total_count):
            if S[A].iloc[row] == value and y.iloc[row] == plus:
                plus_count += 1
            elif S[A].iloc[row] == value and y.iloc[row] == minus:
                minus_count += 1
        s_v = np.array([plus_count, minus_count])
        s_tot += s_v

        attribute_amount = np.sum(S[A]==value)
        sub_gain = (attribute_amount/total_count) * entropy(s_v)
        sub_gains.append(sub_gain)

    total_entropy = entropy(s_tot) - sum(sub_gains)
    return total_entropy

def entropy(counts: np.ndarray):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))




if __name__ == "__main__":
    data_1 = pd.read_csv("decision_tree\data_1.csv")

    # Create and fit a Decrision Tree classifier
    X = data_1.drop(columns=['Play Tennis'])
    y = data_1['Play Tennis']
    y.head()
    temperature = X.columns[3]
    for i in range(len(X.columns)):
        print(X.columns[i])
        print(gain(X, X.columns[i], y))
    # Create and fit a Decrision Tree classifier
    model_1 = DecisionTree()  # <-- Should work with default constructor
    model_1.fit(X, y)

    # Verify that it perfectly fits the training set
    print(f'Accuracy: {accuracy(y_true=y, y_pred=model_1.predict(X)) * 100 :.1f}%')