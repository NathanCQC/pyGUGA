from pytuga import ShavittGraphAB
from scipy.special import binom

def n_csfs(abc):
    n = sum(abc)
    return ((abc[1]+1)/(n+1) )* binom(n +1,abc[0])

def test_shavitt_graph_ab():
    for i in range(1,10):
        abc = (i,0,i)
        guga = ShavittGraphAB(abc)
        csfs = guga._get_csfs()
        print(len(csfs))
        assert len(csfs) == n_csfs(abc)

if __name__ == "__main__":
    test_shavitt_graph_ab()