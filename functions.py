import numpy as np
import matplotlib.pyplot as plt


ROWS    = 6                        # Number of rows in the board
COLS    = 7                        # Number of columns in the board
INPUT   = 2*(ROWS*COLS) + 1        # Number of neurons in the input layer
POPU    = 24                       # Population size ( for the genetic algo )
HIDDEN  = 0                        # Number of hidden layers
LAYERS  = [INPUT, COLS]            # Number of neurons in each layer
COLORS  = {'Red':1, 'Yellow':2}
PLAYERS = ['Draw', 'Red', 'Yellow']


# Board functions

def NewTable():
    return np.zeros(shape=(ROWS,COLS))

def DropIn(table, j, color):
     
    ok = ( not IsColFull(table,j ) )
    for i in range(1,ROWS+1):
        if table[ROWS-i][j]==0:
            table[ROWS-i][j] = color
            break

    return table, ok

def IsFull(board):
    return ( len(board[board==0]) == 0 )

def IsColFull(board, j):
    return ( len(board[:,j][board[:,j]==0]) == 0 )

def DrawBoard(table):
    for i in range(0,ROWS):
        for j in range(0,COLS):
            pedina = table[ROWS-i-1][j]
            if   pedina == 1: color='red'
            elif pedina == 2: color='yellow'
            else : color='white'
            circle = plt.Circle((j,i), 0.4, fc=color)
            plt.gca().add_patch(circle)
    plt.axis('scaled')
    

# Game Evaluation functions ( who won? )

def CheckRow(i, table):
    row = table[i]
    for j in range(0,COLS-4+1):
        x = np.prod(row[j:j+4])
        if x==1**4 :
            return 1     
        elif x==2**4 :
            return 2
    return 0

def CheckColumn(j, table):
    col = table[:,j]
    for i in range(0,ROWS-4+1):
        x = np.prod(col[i:i+4])
        if x==1**4 :
            return 1
        elif x==2**4 :
            return 2
    return 0

def CheckDiagonal(i, j, table, anti=False):
    
    direction = 1 - 2*int(anti)
    diag = np.zeros(4)
    for k in range(0,4):
        diag[k] = table[ i+k, j + direction * k ]
    x = np.prod(diag)

    if x==1**4 :
        return 1
    elif x==2**4 :
        return 2
    else :
        return 0
    
def Winner(table):
    for i in range(0,ROWS):
        x = CheckRow(i, table)
        if x: return x
        
        for j in range(0,COLS):
            x = CheckColumn(j, table)
            if x: return x
            
            if i < ROWS-4+1:
                if j < COLS-4+1:
                    x = CheckDiagonal(i,j, table)
                    if x: return x

                if j >= COLS-4:
                    x = CheckDiagonal(i,j, table, anti=True)
                    if x: return x
    return x

def AnnounceWinner(board):
    x = Winner(board)
    if x > 0: print('{0} won!'.format(PLAYERS[x]))

        
# Neural Net functions

def Sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def HotVector(n):
    v = np.zeros(COLS)
    v[n] = 1
    return v

def RandomTrain(X,Y,N):
    idx = np.random.choice(range(0, len(X)), N)
    return X[idx], Y[idx]

def Input(board):
    X1  = board.reshape(COLS*ROWS).copy()
    X1[X1==2] = 0
    X2  = board.reshape(COLS*ROWS).copy()
    X2[X2==1] = 0   
    X2[X2==2] = 1  
    X  = np.append(1, X1)
    X  = np.append(X, X2)
    return X

def ForwardProp(X0, w):
    X1  = Sigmoid( np.dot( X0, w.T ) )
    return X1

def Response(table, w, full=False):
    
    X0  = Input(table)
    Y   = Sigmoid(ForwardProp(Input(table), w))
    for i in range(0, COLS):
        if ( IsColFull(table,i) ): Y[i] = 0
    
    if full : return Y
    return np.argmax(Y)

def MakeMove(table, w1, p):
    table, ok = DropIn(table, Response(table, w1), p)
    return table


def RandomWeights():
    
    we = []
    for h in range(0, HIDDEN+1):
        we.append( np.random.uniform(low=-1, high=1, size=(LAYERS[h+1], LAYERS[h])) )
    return we


# Random Game functions

def RandomMove(table, moves, x):
    
    ok = False
    while(not ok):
        i = np.random.randint(0, COLS)
        table, ok = DropIn(table,i,1)
    moves.append(i)
    x = Winner(board)
    if x > 0 : return table, moves, x
    
    ok = False  
    while(not ok):
        i = np.random.randint(0, COLS)
        table, ok = DropIn(table,i,2)
    moves.append(i)
    x = Winner(board)
    return table, moves, x

def RandomMatch(table, moves, x):
    for i in range(0,int(ROWS*COLS/2) ):
        table, moves, x = RandomMove(table, moves, x)
        if x > 0 : break
    return table, moves, x


# Allow player 1 to change the last move

def AnotherChance(board, moves, new):
    
    table = board.copy()
    
    j   = moves[-1]
    i   = ROWS - len( np.trim_zeros(table[:,j]) )
    table[i,j] = 0
    
    j   = moves[-2]
    i   = ROWS - len( np.trim_zeros(table[:,j]) ) 
    table[i,j] = 0

    j   = new
    i   = ROWS - len( np.trim_zeros(table[:,j]) ) - 1
    table[i,j] = 1
    
    j   = moves[-1]
    i   = ROWS - len( np.trim_zeros(table[:,j]) ) - 1
    table[i,j] = 2
    
    x = Winner(table)
    moves = moves[:-2] + [ new, moves[-1] ]
    return table, moves, x