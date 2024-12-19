import numpy as np
from typing import List, Tuple

class MerkleTree:
    def __init__(self, arr:np.ndarray, block_size:int = 32) -> None:
        if len(arr.shape) != 2:
            raise ValueError("Merkle Tree array must be 2d")

        self.arr = arr
        self.block_size:int = block_size
        self.shape = np.ceil(np.array(arr.shape)/block_size).astype(int)
        self.height = int(np.max(np.ceil(np.log2(self.shape)))+1)
        self.size:int = int(4**(self.height-1))
        self.tree: np.ndarray = np.zeros(int((4**(self.height)-1)//3), dtype=np.int64)

        self.build(0,self.height, 0, 0)
        self.differences: List[Tuple[int,int,int,int]] = []

    def build(self, node:int, height:int, i0:int, j0:int) -> None:
        self.tree[node] = 0

        if height == 1:
            if i0>=self.arr.shape[0] or j0>=self.arr.shape[1]:
                self.tree[node] = 0
            else:
                i1 = min(i0+self.block_size,self.arr.shape[0])
                j1 = min(j0+self.block_size,self.arr.shape[1])
                self.tree[node] = hash(self.arr[i0:i1,j0:j1].tobytes()) % (1 << 63)

        else:
            for k in range(1,5):
                i = i0
                j = j0
                if k >= 3: i += self.block_size * (1 << (height - 2))
                if k % 2 == 0: j += self.block_size * (1 << (height - 2))
                self.build(4 * node + k, height - 1, i, j)
            self.tree[node] = hash(tuple(self.tree[4*node+i] for i in range(1, 5))) % (1 << 63)

    def compare_aux(self, other: 'MerkleTree', node:int, height:int, i0:int, j0:int) -> None:
        if self.tree[node] == other.tree[node]:
            return

        if height == 1:
            i1 = i0 + self.block_size
            j1 = j0 + self.block_size
            i1 = int(min(i1, self.arr.shape[0]))
            j1 = int(min(j1, self.arr.shape[1]))
            self.differences.append((i0, j0, i1, j1))
            return

        all_different = True
        for k in range(1,5):
            if self.tree[4*node+k] == other.tree[4*node+k]:
                all_different = False
                break

        if all_different:
            i1 = i0 + self.block_size * (1 << (height - 1))
            j1 = j0 + self.block_size * (1 << (height - 1))
            i1 = int(min(i1,self.arr.shape[0]))
            j1 = int(min(j1,self.arr.shape[1]))
            self.differences.append((i0,j0,i1,j1))
            return

        for k in range(1,5):
            if self.tree[4*node+k] != other.tree[4*node+k]:
                i = i0
                j = j0
                if k >= 3: i += self.block_size * (1 << (height - 2))
                if k % 2 == 0: j += self.block_size * (1 << (height - 2))
                self.compare_aux(other, 4*node+k, height-1, int(i), int(j))

    def compare(self, other: 'MerkleTree') -> List[Tuple[int,int,int,int]]:
        """
        Returns the rectangles (i0,j0,i1,j1) which differ between the 2 trees
        """
        self.differences = []
        self.compare_aux(other, 0, self.height, 0 ,0)
        return self.differences


if __name__ == "__main__":
    arr = np.random.random((4,4))
    tree1 = MerkleTree(arr,1)
    arr[0:2,0:2] = 0
    arr[3,3]=0
    tree2 = MerkleTree(arr,1)
    print(tree1.compare(tree2))
