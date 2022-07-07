def print_in_format(matrix):
    for i in range(9):
        if i%3==0 and i>0:
            print("")
        print(str(matrix[i])+" ", end = "")


def linear_to_matrix(index):
    row = int(index / 3)
    column = int(index % 3)
    return row, column

def manhattan(state):
    h = 0

    # Goal
    goal = [1, 2, 3,
            4, 5, 6,
            7, 8, 0]

    for i in range(9):
        if goal[i] != state[i] and state[i] != 0:
            goal_r, goal_c = linear_to_matrix(i)
            state_pos = int(state.index(goal[i]))
            state_r , state_c = linear_to_matrix(state_pos)

            h += abs(goal_r - state_r) + abs(goal_c - state_c)
    
    return h



def move(ar, p, st):
    rh = 9999 # random cost
    store_st = st.copy()
    
    for i in range(len(ar)):
        
        dupl_st = st.copy()
        
        tmp = dupl_st[p]
        dupl_st[p] = dupl_st[arr[i]]
        dupl_st[arr[i]] = tmp
        
        trh = manhattan(dupl_st)
        
        if trh<rh:
            rh = trh
            store_st = dupl_st.copy()   
    return store_st, rh
    
# Problem    
state = [1, 2, 3,
         4, 5, 6,
         8, 7, 0]

h = manhattan(state)
Level = 1

print("\n------ Level "+str(Level)+" ------")
print_in_format(state)
print("\nHeuristic Value(Manhattan Distance) : "+str(h))

# Some states cannot be solved, keep upper limit
u_limit = 50000
while h>0:
    pos = int(state.index(0))
    
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
    