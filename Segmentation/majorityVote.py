from collections import Counter

def majorityVote(votes):
    """
    Parameters
    ----------
        votes:                numpy array

    Return
    ------
        top_two[0][0]:       int
        
    """
    vote_count = Counter(votes)
    top_two = vote_count.most_common(1)

    return top_two[0][0]
    

if __name__ == "__main__":

    import numpy as np
    votes = np.array([1,1,1,2,2,2])
    print((majorityVote(votes)))

