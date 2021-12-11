def recursive_fact(x):
    if x <= 1:
        return 1
    return x * recursive_fact(x - 1)

def iterative_fact(x):
    i, j = 1, 1
    while i <= x:
        i, j = i + 1, j * i

    return j

print(recursive_fact(5))
print(iterative_fact(5))

assert recursive_fact(5) == iterative_fact(5)


def num_eights(pos):
    """Returns the number of times 8 appears as a digit of pos."""

    if pos:
        return (1 if pos % 10 == 8 else 0) + num_eights(pos // 10)
    else:
        return 0

print(num_eights(851803180801223238358))

def make_anonymous_factorial():
    """Return the value of an expression that computes factorial."""

    # ??? wtf is a combinator
    return (lambda f: (lambda x: f(f, x)))(lambda g, n: 1 if n == 0 else n * g(g, n - 1))

print(make_anonymous_factorial()(5))

def ascending_coin(coin):
    """Returns the next ascending coin in order."""
    if coin == 1:
        return 5
    elif coin == 5:
        return 10
    elif coin == 10:
        return 25


def descending_coin(coin):
    """Returns the next descending coin in order."""
    if coin == 25:
        return 10
    elif coin == 10:
        return 5
    elif coin == 5:
        return 1


def count_coins(change):
    """Return the number of ways to make change using coins of value of 1, 5, 10, 25.
    >>> count_coins(100) # How many ways to make change for a dollar?
    242
    >>> count_coins(200)
    1463
    """
    def helper(amount_remaining, coin):
        if amount_remaining == 0:
            return 1
        if amount_remaining < 0:
            return 0
        if coin == None:
            return 0
        with_coin = helper(amount_remaining - coin, coin)
        without_coin = helper(amount_remaining, descending_coin(coin))
        return with_coin + without_coin
    return helper(change, 25)

# print(count_coins(100))
# print(count_coins(200))