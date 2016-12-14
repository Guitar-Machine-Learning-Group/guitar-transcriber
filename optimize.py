import os
import numpy as np
from sklearn.metrics import f1_score

def optimizer( array, size = 2 ):

    row, col = np.shape( array )

    for i in range( 1, row - size - 1 ):

        for j in range( col ):

            if array[i-1,j] == array[i+size+1,j]:
                array[i:i+size,j] = array[i-1,j]

    return array

            

            

if __name__ == '__main__':
    fname = "3doorsdown_herewithoutyou_rnn_80_pred.npy"
    oname = os.getcwd() + fname
    truth = np.load( "3doorsdown_herewithoutyou.npy" )
    mine  = np.load( fname )
    f1    = [ f1_score( mine, truth, average = 'micro' ) ]

    for i in range( 1, 11 ):

        f1.append( f1_score( optimizer( mine, i ), truth, average = 'micro' ) )

    print( f1.index(max(f1)) )
    print( f1 )

    print( fname + " comeplete" )
