import numpy as np
from typing import List, Tuple

from solutions.mapper.utils import Grid # type: ignore


class MerkleTree:

    def __init__(self, confidence:Grid, occupancy:Grid,  block_size:int = 32) -> None:

        self.arr_shape = confidence.get_grid().shape
        if len(self.arr_shape) != 2:
            raise ValueError("Merkle Tree array must be 2d")

        self.confidence = confidence
        self.occupancy = occupancy
        self.block_size = block_size

        self.shape = np.ceil(np.array(confidence.get_grid().shape) / block_size).astype(int)
        self.height = int(np.max(np.ceil(np.log2(self.shape)))+1)
        self.size = int(4**(self.height-1))
        self.tree = np.zeros(int((4**(self.height)-1)//3), dtype=np.int64)

        self.build(0,self.height, 0, 0)
        self.differences = []

    def build(self, node, height, i0, j0):
        self.tree[node] = 0

        if height == 1:
            if i0>=self.arr_shape[0] or j0>=self.arr_shape[1]:
                self.tree[node] = 0
            else:
                i1 = min(i0 + self.block_size, self.arr_shape[0])
                j1 = min(j0 + self.block_size, self.arr_shape[1])
                self.tree[node] = hash(self.confidence.get_grid()[i0:i1, j0:j1].tobytes()) % (1 << 63)

        else:
            for k in range(1,5):
                i = i0
                j = j0
                if k >= 3: i += self.block_size * (1 << (height - 2))
                if k % 2 == 0: j += self.block_size * (1 << (height - 2))
                self.build(4 * node + k, height - 1, i, j)
            self.tree[node] = hash(tuple(self.tree[4*node+i] for i in range(1, 5))) % (1 << 63)

    def compare_aux(self, other, node, height, i0, j0):
        if self.tree[node] == other.tree[node]:
            return

        if height == 1:
            i1 = i0 + self.block_size
            j1 = j0 + self.block_size
            i1 = int(min(i1, self.arr_shape[0]))
            j1 = int(min(j1, self.arr_shape[1]))
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
            i1 = int(min(i1, self.arr_shape[0]))
            j1 = int(min(j1, self.arr_shape[1]))
            self.differences.append((i0,j0,i1,j1))
            return

        for k in range(1,5):
            if self.tree[4*node+k] != other.tree[4*node+k]:
                i = i0
                j = j0
                if k >= 3: i += self.block_size * (1 << (height - 2))
                if k % 2 == 0: j += self.block_size * (1 << (height - 2))
                self.compare_aux(other, 4*node+k, height-1, int(i), int(j))

    def compare(self, other):
        """
        Returns the rectangles (i0,j0,i1,j1) which differ between the 2 trees
        """
        self.differences = []
        self.compare_aux(other, 0, self.height, 0 ,0)
        return self.differences


    def update_aux(self, node, height, curr_rect, rect):
        if height == 1:
            i = min(curr_rect[0] + self.block_size, self.arr_shape[0])
            j = min(curr_rect[1] + self.block_size, self.arr_shape[1])
            self.tree[node] = hash(self.confidence.get_grid()[curr_rect[0]:i, curr_rect[1]:j].tobytes()) % (1 << 63)
            return

        for k in range(1, 5):
            x0,x1 = curr_rect[0],curr_rect[2]
            y0,y1 = curr_rect[1],curr_rect[3]
            if k >= 3:
                x0 += self.block_size * (1 << (height - 2))
            else:
                x1 -= self.block_size * (1 << (height - 2))
            if k % 2 == 0:
                y0 += self.block_size * (1 << (height - 2))
            else:
                y1 -= self.block_size * (1 << (height - 2))

            if (max(x0, rect[0]) <= min(x1, rect[2])) and (max(y0, rect[1]) <= min(y1, rect[3])):
                self.update_aux(4 * node + k, height - 1, (x0,y0,x1,y1), rect)

        self.tree[node] = hash(tuple(self.tree[4 * node + i] for i in range(1, 5))) % (1 << 63)


    def update(self, i0, j0, i1, j1):
        """
        updates the merkle tree in the rect (inclusive) specified
        """
        self.update_aux(0,
                        self.height,
                        (0, 0, self.block_size*(1<<(self.height-1))-1, self.block_size*(1<<(self.height-1))-1),
                        (i0, j0, i1, j1))


    def merge(self, other: 'MerkleTree') -> None:
        differences:List[Tuple[int,int,int,int]] = self.compare(other)
        for i0,j0,i1,j1 in differences:
            self.occupancy.get_grid()[i0:i1, j0:j1] = np.where(self.confidence.get_grid()[i0:i1, j0:j1] > other.confidence.get_grid()[i0:i1, j0:j1], self.occupancy.get_grid()[i0:i1, j0:j1], other.occupancy.get_grid()[i0:i1, j0:j1])
            self.confidence.get_grid()[i0:i1,j0:j1] = np.maximum(self.confidence.get_grid()[i0:i1,j0:j1], other.confidence.get_grid()[i0:i1,j0:j1])
            self.update(i0, j0, i1, j1)


if __name__ == "__main__":
    confidence1 = Grid((5,5), 1)
    occupancy1 = Grid((5,5), 1)
    tree1 = MerkleTree(confidence1, occupancy1, 1)
    print(tree1.occupancy.get_grid())

    confidence2 = Grid((5, 5), 1)
    occupancy2 = Grid((5, 5), 1)
    confidence2.get_grid()[4,4] = 1
    occupancy2.get_grid()[4, 4] = 1
    tree2 = MerkleTree(confidence2, occupancy2, 1)
    tree1.merge(tree2)
    print("After update:")
    print(tree1.occupancy.get_grid())
