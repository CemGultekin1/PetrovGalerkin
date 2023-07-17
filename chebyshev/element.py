

import itertools


class Degree:
    def __init__(self,degree:int) -> None:
        self.degree = degree
    def degree_index_product(self,prod:int):
        return itertools.product(*[range(self.degree) for _ in range(prod)])