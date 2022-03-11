def recursive_fact(x):
    if x <= 1:
        return 1
    return x * recursive_fact(x - 1)
    
print(recursive_fact(5))