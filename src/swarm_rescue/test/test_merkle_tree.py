import os

import numpy as np
from typing import Optional, Any

#import pytest

from solutions.mapper.utils import Grid, MerkleTree # type: ignore
from solutions.mapper.merkle_tree import MerkleTree as MerklePython # type: ignore


def test_build() -> None:
    for n in range(1,100):
        for size in range(1,20):
            confidence = Grid((n,n),1)
            occupancy = Grid((n,n),1)
            tree = MerkleTree(confidence,occupancy,size)

def test_update() -> None:
    for n in [4,7,8,9,10,15,16,17,20,22,31,32,60,64,100,110,128]:
        for size in range(1,4):
            confidence1 = Grid((n, n), 1)
            occupancy1 = Grid((n, n), 1)
            tree1 = MerkleTree(confidence1,occupancy1,size)

            tree_copy = tree1.tree.copy()
            confidence1.get_grid()[0,0] = 1
            tree1.update(0,0,0,0)

            assert tree1.tree[0]!=tree_copy[0] and  tree1.tree[1]!=tree_copy[1], f"For shape ({n},{n}) and block size {size}"
            assert tree1.tree[2]==tree_copy[2] and tree1.tree[3]==tree_copy[3] and tree1.tree[4]==tree_copy[4], f"For shape ({n},{n}) and block size {size}"

def test_update2() -> None:
    for n in [4, 7, 8, 9, 10, 15, 16, 17, 20, 22, 31, 32, 60, 64, 100, 110, 128]:
        for size in range(1, 4):
            confidence1 = Grid((n, n), 1)
            occupancy1 = Grid((n, n), 1)
            tree1 = MerkleTree(confidence1, occupancy1, size)

            tree_copy = tree1.tree.copy()
            confidence1.get_grid()[0, n-1] = 1
            confidence1.get_grid()[n-1, n-1] = 1
            tree1.update(0, 0, n-1, n-1)

            assert tree1.tree[0] != tree_copy[0] and tree1.tree[2] != tree_copy[2] and tree1.tree[4] != tree_copy[4], \
                f"For shape ({n},{n}) and block size {size}"
            assert tree1.tree[1] == tree_copy[1] and tree1.tree[3] == tree_copy[3], \
                f"For shape ({n},{n}) and block size {size}"

def test_update3() -> None:
    for n in [4, 7, 8, 9, 10, 15, 16, 17, 20, 22, 31, 32, 60, 64, 100, 110, 128]:
        for size in range(1, 4):
            confidence1 = Grid((n, n), 1)
            occupancy1 = Grid((n, n), 1)
            tree1 = MerkleTree(confidence1, occupancy1, size)

            tree_copy = tree1.tree.copy()
            confidence1.get_grid()[0, n-1] = 1
            confidence1.get_grid()[n-1, n-1] = 1
            tree1.update(n//2, n//2, n-1, n-1)
            tree1.update(0, n//2, n//2, n-1)

            assert tree1.tree[0] != tree_copy[0] and tree1.tree[2] != tree_copy[2] and tree1.tree[4] != tree_copy[4], \
                f"For shape ({n},{n}) and block size {size}"
            assert tree1.tree[1] == tree_copy[1] and tree1.tree[3] == tree_copy[3], \
                f"For shape ({n},{n}) and block size {size}"
