# ast-experiments

My CS61A (Structure and Interpretation of Computer Programs) plus project and project shown at the project showcase for the Fall 2021 version of the course taught by John Denero and Pamela Fox.
The main component of the project was a Python metacircular evaluator.

## Example output
Running `python3.10 ast_evaluator.py --visualize visualize_test.py` (shown below)
```python
def recursive_fact(x):
    if x <= 1:
        return 1
    return x * recursive_fact(x - 1)
    
print(recursive_fact(5))
```
<details><summary>outputs (output truncated, click triangle to expand)</summary>
<p>

```python
Python AST (depth 2)
Compare(
    lineno=2,
    col_offset=7,
    end_lineno=2,
    end_col_offset=13,
    left=Name(lineno=2, col_offset=7, end_lineno=2, end_col_offset=8, id='x', ctx=Load()),
    ops=[LtE()],
    comparators=[Constant(lineno=2, col_offset=12, end_lineno=2, end_col_offset=13, value=1, kind=None)],
)
--------------------------------------------------
Python Source
Line number 2, columns 7 to 13
    if x <= 1:
       ˜˜˜˜˜˜
--------------------------------------------------
Result
True
--------------------------------------------------
Python AST (depth 2)
Constant(lineno=3, col_offset=15, end_lineno=3, end_col_offset=16, value=1, kind=None)
--------------------------------------------------
Python Source
Line number 3, columns 15 to 16
        return 1
               ˜
--------------------------------------------------
Result
1
--------------------------------------------------
Python AST (depth 2)
Call(
    lineno=4,
    col_offset=15,
    end_lineno=4,
    end_col_offset=36,
    func=Name(lineno=4, col_offset=15, end_lineno=4, end_col_offset=29, id='recursive_fact', ctx=Load()),
    args=[
        BinOp(
            lineno=4,
            col_offset=30,
            end_lineno=4,
            end_col_offset=35,
            left=...,
            op=...,
            right=...,
        ),
    ],
    keywords=[],
)
--------------------------------------------------
Python Source
Line number 4, columns 15 to 36
    return x * recursive_fact(x - 1)
               ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
1
--------------------------------------------------
Python AST (depth 2)
BinOp(
    lineno=4,
    col_offset=11,
    end_lineno=4,
    end_col_offset=36,
    left=Name(lineno=4, col_offset=11, end_lineno=4, end_col_offset=12, id='x', ctx=Load()),
    op=Mult(),
    right=Call(
        lineno=4,
        col_offset=15,
        end_lineno=4,
        end_col_offset=36,
        func=...,
        args=[
            ...,
        ],
        keywords=[],
    ),
)
--------------------------------------------------
Python Source
Line number 4, columns 11 to 36
    return x * recursive_fact(x - 1)
           ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
2
--------------------------------------------------
Python AST (depth 2)
Call(
    lineno=4,
    col_offset=15,
    end_lineno=4,
    end_col_offset=36,
    func=Name(lineno=4, col_offset=15, end_lineno=4, end_col_offset=29, id='recursive_fact', ctx=Load()),
    args=[
        BinOp(
            lineno=4,
            col_offset=30,
            end_lineno=4,
            end_col_offset=35,
            left=...,
            op=...,
            right=...,
        ),
    ],
    keywords=[],
)
--------------------------------------------------
Python Source
Line number 4, columns 15 to 36
    return x * recursive_fact(x - 1)
               ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
2
--------------------------------------------------
Python AST (depth 2)
BinOp(
    lineno=4,
    col_offset=11,
    end_lineno=4,
    end_col_offset=36,
    left=Name(lineno=4, col_offset=11, end_lineno=4, end_col_offset=12, id='x', ctx=Load()),
    op=Mult(),
    right=Call(
        lineno=4,
        col_offset=15,
        end_lineno=4,
        end_col_offset=36,
        func=...,
        args=[
            ...,
        ],
        keywords=[],
    ),
)
--------------------------------------------------
Python Source
Line number 4, columns 11 to 36
    return x * recursive_fact(x - 1)
           ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
6
--------------------------------------------------
Python AST (depth 2)
Call(
    lineno=4,
    col_offset=15,
    end_lineno=4,
    end_col_offset=36,
    func=Name(lineno=4, col_offset=15, end_lineno=4, end_col_offset=29, id='recursive_fact', ctx=Load()),
    args=[
        BinOp(
            lineno=4,
            col_offset=30,
            end_lineno=4,
            end_col_offset=35,
            left=...,
            op=...,
            right=...,
        ),
    ],
    keywords=[],
)
--------------------------------------------------
Python Source
Line number 4, columns 15 to 36
    return x * recursive_fact(x - 1)
               ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
6
--------------------------------------------------
Python AST (depth 2)
BinOp(
    lineno=4,
    col_offset=11,
    end_lineno=4,
    end_col_offset=36,
    left=Name(lineno=4, col_offset=11, end_lineno=4, end_col_offset=12, id='x', ctx=Load()),
    op=Mult(),
    right=Call(
        lineno=4,
        col_offset=15,
        end_lineno=4,
        end_col_offset=36,
        func=...,
        args=[
            ...,
        ],
        keywords=[],
    ),
)
--------------------------------------------------
Python Source
Line number 4, columns 11 to 36
    return x * recursive_fact(x - 1)
           ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
24
--------------------------------------------------
Python AST (depth 2)
Call(
    lineno=4,
    col_offset=15,
    end_lineno=4,
    end_col_offset=36,
    func=Name(lineno=4, col_offset=15, end_lineno=4, end_col_offset=29, id='recursive_fact', ctx=Load()),
    args=[
        BinOp(
            lineno=4,
            col_offset=30,
            end_lineno=4,
            end_col_offset=35,
            left=...,
            op=...,
            right=...,
        ),
    ],
    keywords=[],
)
--------------------------------------------------
Python Source
Line number 4, columns 15 to 36
    return x * recursive_fact(x - 1)
               ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
24
--------------------------------------------------
Python AST (depth 2)
BinOp(
    lineno=4,
    col_offset=11,
    end_lineno=4,
    end_col_offset=36,
    left=Name(lineno=4, col_offset=11, end_lineno=4, end_col_offset=12, id='x', ctx=Load()),
    op=Mult(),
    right=Call(
        lineno=4,
        col_offset=15,
        end_lineno=4,
        end_col_offset=36,
        func=...,
        args=[
            ...,
        ],
        keywords=[],
    ),
)
--------------------------------------------------
Python Source
Line number 4, columns 11 to 36
    return x * recursive_fact(x - 1)
           ˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜˜
--------------------------------------------------
Result
120
--------------------------------------------------
```

</p>
</details>

## Usage
Unfortunately, this project does not contain all of the code needed to actually output any results.
The interpreter is missing frame.py, and the transpiler is omitted entirely. Both contain code which was extremely similar to the final project for the class, 
so I judged that it was best to leave them out. 

However, if you are a future CS61A student and can supply your own implementation of frame.py (very similar to the one used in the Scheme interpreter, but with support for Python builtin modules
using `builtins.__dict__.items()`), then it should work.

In that case, you can either use it as a REPL with the `--repl` flag or pass in a Python file as input on the command line. If you want a visualization of what the evaluator is doing, you can 
also turn on the `--visualize` flag.

## Explanation
For a video explanation, click here:
https://youtu.be/EuQZW9qEcTg

## Limitations
Only a very small subset of the Python language is implemented. In particular, it is able to evaluate: 

- lambda expressions
- pass statements
- assert statements
- delete statements 
- augmented assignment statements (e.g. `x += 1`)
- object member access ("dot notation")
- if statements
- dictionary literals
- set literals
- list literals
- tuple literals
- binary operations
- unary operations
- return statements
- extended and comparisons (e.g. `1 <= x <= 3`)
- constants
- break statements
- continue statements
- while loops
- variable assignment
- function calls
- function definitions

Nonetheless, this is enough to run many of the functional programming examples that are explored in the first few weeks of CS61A, which is pretty neat.

#### Control Statement Implementation
For me, implementing the `break` and `continue` statements for flow control was one of the most interesting parts of this project. The challenge with both
of these statements is that given how this evaluator is written, the calls to `evaluate_ast` which correspond to the loop that is being broken out of are almost certainly
several call frames "above" the call that is actually doing the `break` or `continue` (after all, when does it really make sense to unconditionally `break` out of a loop?).
The solution I landed on was using Python's exception system to break the normal control flow and jump out of multiple call frames by only installing exception handlers in 
the functions which handle looping. 
