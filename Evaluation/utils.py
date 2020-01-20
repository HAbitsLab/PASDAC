import sys

def lprint(logfile, *argv): # for python version 3

    """ 
    Function description: 
    ----------
        Save output to log files and print on the screen.
    

    Function description: 
    ----------
        var = 1
        lprint('log.txt', var)
        lprint('log.txt','Python',' code')


    Parameters
    ----------
        logfile:                 the log file path and file name.
        argv:                    what should 


    Return
    ------
        none

    Author
    ------
    Shibo(shibozhang2015@u.northwestern.edu)
    """

    # argument check
    if len(argv) == 0:
        print('Err: wrong usage of func lprint().')
        sys.exit()

    argAll = argv[0] if isinstance(argv[0], str) else str(argv[0])
    for arg in argv[1:]:
    	argAll = argAll + (arg if isinstance(arg, str) else str(arg))
    
    print(argAll)

    if logfile != None:
        with open(logfile, 'a') as out:
            out.write(argAll + '\n')


if __name__ == "__main__":
    
    # 5 good test cases
    var = 1
    lprint('log.txt', var)
    lprint('log.txt', 'var is ', var)
    lprint('log.txt', 'Python', ' code')
    lprint(None,var)
    lprint(None,'log')

    # # error catching test case 
    # lprint(var)
    lprint('log.txt')
