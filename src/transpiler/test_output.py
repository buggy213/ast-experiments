
from pair import *

def recursive_fact(n):
    if (n <= 1):1
    else:(n * recursive-fact((n - 1)))

def tail_fact(n):

    def fact_iter(x, y):
        if (x > n):y
        else:fact_iter((x + 1), (x * y))fact_iter(1, 1)recursive_fact(5)tail_fact(5)

