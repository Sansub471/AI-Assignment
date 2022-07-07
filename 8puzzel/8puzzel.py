def print_in_format(puzzel_matrix):
    for i in range(9):
        if i%3==0 and i>0:
            print("")
        print(str(puzzel_matrix[i])+" ", end = "")

def count_misplaced(s):
    count = 0
    ideal = [1, 2, 3,
             4, 5, 6,
             7, 8, 0]
    
    for i in range(9):
        if s[i]!=0 and s[i]!=ideal[i]:
            count+=1
    return count


def move(ar, p, st):
    rh = 99999 # Random high cost
    store_st = st.copy()
    
    for i in range(len(ar)):
        
        dupl_st = st.copy()
        
        tmp = dupl_st[p]
        dupl_st[p] = dupl_st[arr[i]]
        dupl_st[arr[i]] = tmp
        
        trh = count_misplaced(dupl_st)
        
        if trh<rh:
            rh = trh
            store_st = dupl_st.copy()
    
    #print(rh, store_st)
    
    return store_st, rh
    
# Give the problem  
state = [4, 2, 5,
         1, 3, 6,
         8, 0, 7]

h = count_misplaced(state)
Level = 1

print("\n------ Level "+str(Level)+" ------")
print_in_format(state)
print("\nHeuristic Value(Misplaced) : "+str(h))

# Some states cannot be solved, keep upper limit
u_limit = 50000
while h>0:
    pos = int(state.index(0)) # get vacant position
    
    Level += 1
    
    if pos==0:
        arr = [1, 3]
        state, h = move(arr, pos, state)
    elif pos==1:
        arr = [0, 2, 4]
        state, h = move(arr, pos, state)
    elif pos==2:
        arr = [1, 5]
        state, h = move(arr, pos, state)
    elif pos==3:
        arr = [0, 4, 6]
        state, h = move(arr, pos, state)
    elif pos==4:
        arr = [1, 3, 5, 7]
        state, h = move(arr, pos, state)
    elif pos==5:
        arr = [2, 4, 8]
        state, h = move(arr, pos, state)
    elif pos==6:
        arr = [3, 7]
        state, h = move(arr, pos, state)
    elif pos==7:
        arr = [4, 6, 8]
        state, h = move(arr, pos, state)
    elif pos==8:
        arr = [5, 6]
        state, h = move(arr, pos, state)

    if Level < u_limit:  
        print("\n------ Level "+str(Level)+" ------")
        print_in_format(state)
        print("\nHeuristic Value(Misplaced) : "+str(h))
    else:
        print("Could not be solved")
        break