import numpy as np


class MerkleTree:

    def __init__(self, arr:np.array, block_size:int = 32):
        if len(arr.shape) != 2:
            raise ValueError("Merkle Tree array must be 2d")

        self.arr = arr
        self.block_size:int = block_size
        self.shape = np.ceil(np.array(arr.shape)/block_size).astype(int)
        self.height = np.max(np.ceil(np.log2(self.shape+1)))
        self.size:int = int(4**(self.height-1))
        self.tree: np.ndarray = np.zeros(int((4**(self.height)-1)//3), dtype=int)

        self.build(0,self.height)

    def build(self, node:int, level:int):
        self.tree[node] = 0

        if level == 1:
            x = node - len(self.tree) + self.size
            i0 = x // self.size
            j0 = x % self.size
            if i0>=self.arr.shape[0] or j0>=self.arr.shape[1]:
                self.tree[node] = 0
            else:
                i1 = min(i0+self.block_size,self.arr.shape[0])
                j1 = min(j0+self.block_size,self.arr.shape[1])
                self.tree[node] = hash(self.arr[i0:i1,j0:j1].tobytes()) % (1 << 31)

        else:
            for i in range(1,5):
                self.build(4*node+i, level-1)
            self.tree[node] = hash((self.tree[4*node+i] for i in range(1, 5))) % (1 << 31)


if __name__ == "__main__":
    tree = MerkleTree(np.random.random((10,10)),2)
    print("ok")