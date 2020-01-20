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
    votes = np.array([1,1,3,4,2,2,2,2,2,1,1,0])
    print((majorityVote(votes)))

