
# Dynamic Programming implementation of LCS problem  
# Returns length of LCS for X[0..m-1], Y[0..n-1]  
def lcs(X, Y): 
    m = len(X)
    n = len(Y)
    L = [[0 for y in range(m+1)] for x in range(n+1)] 
    # Following steps build L[m+1][n+1] in bottom up fashion. Note 
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]  
    for i in range(n+1): 
        for j in range(m+1): 
            if i == 0 or j == 0: 
                L[i][j] = 0
            elif X[j-1] == Y[i-1]: 
                L[i][j] = L[i-1][j-1] + 1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
    
    # Following code is used to print LCS 
    index = L[n][m] 
    # Create a character array to store the lcs string 
    lcs = [""] * (index+1) 
    lcs[index] = "" 
    # Start from the right-most-bottom-most corner and 
    # one by one store characters in lcs[] 
    i = n 
    j = m 
    while i > 0 and j > 0: 
        # If current character in X[] and Y are same, then 
        # current character is part of LCS 
        if X[j-1] == Y[i-1]: 
            lcs[index-1] = X[j-1] 
            i-=1
            j-=1
            index-=1
        # If not same, then find the larger of two and 
        # go in the direction of larger value 
        elif L[i-1][j] > L[i][j-1]: 
            i-=1
        else: 
            j-=1
    print ("
LCS of " + X + " and " + Y + " is " + "".join(lcs))
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[n][m] 
#end of function lcs 
  
# Driver program to test the above function 
X = AGGTAB
Y = GXTXAYB
print ("
Length of LCS is ", lcs(X, Y) )
