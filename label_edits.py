import numpy as np 

def edit_distance(sent1, sent2, max_id=4999):
    # edit from sent1 to sent2
    # Create a table to store results of subproblems
    m = len(sent1)
    n = len(sent2)
    dp = [[0 for x in range(n+1)] for x in range(m+1)]
    # Fill d[][] in bottom up manner
    for i in range(m+1):
        for j in range(n+1):
            # If first string is empty, only option is to
            # isnert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif sent1[i-1] == sent2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            # If last character are different, consider all
            # possibilities and find minimum
            else:
                edit_candidates = np.array([
                    dp[i][j-1], # Insert
                    dp[i-1][j] # Remove
                    ])
                dp[i][j] = 1 + min(edit_candidates)
    return dp

def sent2edit(sent1, sent2):
    # print(sent1,sent2)
    '''
    '''
    dp = edit_distance(sent1, sent2)
    edits = []
    pos = []
    m, n = len(sent1), len(sent2)
    while m != 0 or n != 0:
        curr = dp[m][n]
        if m==0: #have to insert all here
            while n>0:
                left = dp[1][n-1]
                edits.append(sent2[n-1])
                pos.append(left)
                n-=1
        elif n==0:
            while m>0:
                top = dp[m-1][n]
                edits.append('DEL')
                pos.append(top)
                m -=1
        else: # we didn't reach any special cases yet
            diag = dp[m-1][n-1]
            left = dp[m][n-1]
            top = dp[m-1][n]
            if sent2[n-1] == sent1[m-1]: # keep
                edits.append('KEEP')
                pos.append(diag)
                m -= 1
                n -= 1
            elif curr == top+1: # INSERT preferred before DEL
                edits.append('DEL')
                pos.append(top)  # (sent2[n-1])
                m -= 1
            else: #insert
                edits.append(sent2[n - 1])
                pos.append(left)  # (sent2[n-1])
                n -= 1
    edits = edits[::-1]
    # replace the keeps at the end to stop, this helps a bit with imbalanced classes (KEEP,INS,DEL,STOP)
    for i in range(len(edits))[::-1]: #reversely checking
        if edits[i] == 'KEEP':
            if edits[i-1] =='KEEP':
                edits.pop(i)
            else:
                edits[i] = 'STOP'
                break
    # if edits == []: # do we learn edits if input and output are the same?
    #     edits.append('STOP') #in the case that input and output sentences are the same
    return edits


def edit2sent(sent, edits, last=False):
    """
    Edit the sentence given the edit operations.
    :param sent: sentence to edit, list of string
    :param edits: a sequence of edits in ['KEEP','DEL','STOP']+INS_vocab_set
    :return: the new sentence, as the edit sequence is deterministic based on the edits labels
    """
    new_sent = []
    sent_pointer = 0 #counter the total of KEEP and DEL, then align with original sentence

    if len(edits) == 0 or len(sent) ==0: # edit_list empty, return original sent
        return sent

    for i, edit in enumerate(edits):
        if len(sent) > sent_pointer: #there are tokens left for editing
            if edit =="KEEP":
                new_sent.append(sent[sent_pointer])
                sent_pointer += 1
            elif edit =="DEL":
                sent_pointer += 1
            elif edit == 'STOP':
                break # go outside the loop and copy everything after current sent_pointer into the new sentence
            else: #insert the word in
                new_sent.append(edit)

    if sent_pointer < len(sent):
        for i in range(sent_pointer,len(sent)):
            new_sent.append(sent[i])
    return new_sent

def decode_edit(p_pos, p_edit, p_wins, p_wsub):
    '''
    Edit the sentence given the prediction of the model
    '''
    bsz = p_wins.shape[0]
    
    edit = np.argmax(p_edit, axis=-1)
    pos  = np.argmax(p_pos, axis=-1)
    wins = np.argmax(p_wins, axis=-1)[np.arange(bsz), pos]
    wsub = np.argmax(p_wsub, axis=-1)[np.arange(bsz), pos]
    #print(edit.shape, pos.shape, wins.shape, wsub.shape)
    return edit, pos, wins, wsub

def main():
    #this prints an example of doing minimum editing distance
    sent1 = "Military experts say the line between combat is getting blurry .".split()
    sent2 = "Military experts say war is changing .".split()

    A = edit_distance(sent1, sent2, max_id=4999)
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
          for row in A]))

    B = sent2edit(sent1, sent2)
    print(B)
    print(edit2sent(sent1,B))

if __name__ == '__main__':
    main()