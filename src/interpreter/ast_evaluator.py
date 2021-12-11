import ast
import argparse
import sys
from types import FunctionType, GeneratorType
from typing import Any, List, Type

import astpretty
from frame import *


# this is only tested to work with Python 3.8.10

class AstFunction():
    def __init__(self, body: List[ast.AST], arguments_descriptor: ast.arguments, 
                frame: Frame, interpreter, name: str, lineno: int, is_lambda=False) -> None:
        """Represents a user-defined function. Encodes information about the function signature, lexical scope, default parameters, etc."""
        self.body = body
        self.arguments_descriptor = arguments_descriptor
        self.frame = frame
        self.interpreter = interpreter
        self.eval_function = interpreter.evaluate_ast
        self.is_lambda = is_lambda
        self.name = name
        self.lineno = lineno

        # default values are only evaluated once
        def curried_evaluator(node): return None if node is None else self.eval_function(
            node, frame)

        def get_arg(x): return x.arg

        # this is really horrible but basically it handles positional only, normal argument, and keyword argument default values
        self.defaults = dict(zip(list(map(get_arg, arguments_descriptor.posonlyargs + arguments_descriptor.args))[-len(arguments_descriptor.defaults):],
            map(curried_evaluator, arguments_descriptor.defaults)))
        self.kwdefaults = {k: v for k, v in zip(map(get_arg, arguments_descriptor.kwonlyargs), map(
            curried_evaluator, arguments_descriptor.kw_defaults))}

    def validate_parameters(self, args, kwargs):
        """
        Ensure that the parameters passed into this function match the function signature ("arguments descriptor")
        """
        for i, kwonlyarg in enumerate(self.arguments_descriptor.kwonlyargs):
            if kwonlyarg.arg in kwargs:
                continue
            if self.arguments_descriptor.kw_defaults[i] is None:
                raise TypeError(
                    f'expected keyword argument {kwonlyarg.arg} not given')

        for kwarg in kwargs:
            def get_arg(x): return x.arg
            if kwarg not in map(get_arg, self.arguments_descriptor.kwonlyargs) \
                    and kwarg not in map(get_arg, self.arguments_descriptor.args):
                raise TypeError(f'unexpected keyword argument {kwarg}')

        if len(args) > len(self.arguments_descriptor.posonlyargs) + len(self.arguments_descriptor.args):
            raise TypeError(f'unexpected positional arguments given')

        expected_arguments = len(self.arguments_descriptor.posonlyargs) + len(
            self.arguments_descriptor.args) - len(self.arguments_descriptor.defaults)
        if len(args) < expected_arguments:
            raise TypeError(
                f'expected {expected_arguments} positional arguments, got {len(args)}')

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.validate_parameters(args, kwds)

        args = {**self.defaults, **dict(zip(map(lambda x: x.arg, (self.arguments_descriptor.posonlyargs + self.arguments_descriptor.args)[:len(args)]),
                args))}
        kwds = {**self.kwdefaults, **kwds}

        new_frame = self.frame.create_child_frame(self.arguments_descriptor, args, kwds)
        self.interpreter.push_frame(new_frame)

        # lambdas are guaranteed to return an expression
        if self.is_lambda:
            return self.eval_function(self.body, new_frame)
        
        try:
            for statement in self.body:
                self.eval_function(statement, new_frame)
        except ReturnSignal as signal:
            return signal.returned_value
    
    def __repr__(self):
        return f'<{self.name} function, line {self.lineno}>'

# signals for control flow statements
class ContinueSignal(Exception):
    pass

class BreakSignal(Exception):
    pass

class ReturnSignal(Exception):
    def __init__(self, returned_value) -> None:
        self.returned_value = returned_value

# all possible binary operations
BINARY_OPERATIONS = {
    ast.Add: lambda l, r: l + r,
    ast.Sub: lambda l, r: l - r,
    ast.Mult: lambda l, r: l * r,
    ast.Div: lambda l, r: l / r,
    ast.FloorDiv: lambda l, r: l // r,
    ast.Mod: lambda l, r: l % r,
    ast.Pow: lambda l, r: l ** r,
    ast.LShift: lambda l, r: l << r,
    ast.RShift: lambda l, r: l >> r,
    ast.BitOr: lambda l, r: l | r,
    ast.BitXor: lambda l, r: l ^ r,
    ast.BitAnd: lambda l, r: l & r,
    ast.MatMult: lambda l, r: l @ r
}

# all possible unary operations
UNARY_OPERATIONS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
    ast.Not: lambda x: not x,
    ast.Invert: lambda x: ~x
}

# all possible comparison operations
COMPARISON_OPERATIONS = {
    ast.Eq: lambda l, r: l == r,
    ast.NotEq: lambda l, r: l != r,
    ast.Lt: lambda l, r: l < r,
    ast.LtE: lambda l, r: l <= r,
    ast.Gt: lambda l, r: l > r,
    ast.GtE: lambda l, r: l >= r,
    ast.Is: lambda l, r: l is r,
    ast.IsNot: lambda l, r: l is not r,
    ast.In: lambda l, r: l in r,
    ast.NotIn: lambda l, r: l not in r,
}

class AstInterpreter():

    def __init__(self, repl=False) -> None:
        self.frames = [global_frame]
        self.repl = repl

    def push_frame(self, frame):
        self.frames.append(frame)

    def pop_frame(self):
        if len(self.frames) > 1:
            self.frames.pop()
        else:
            raise RuntimeError('cannot pop global frame')

    def evaluate_ast(self, node: ast.AST, frame: Frame, **kwargs) -> Any:
        dispatch_dict = {}

        # Utility functions
        def dispatch(node_type: Type[ast.AST]) -> FunctionType:
            """Decorator to add functions which handle a given AST node type to the dispatch_dict."""
            def decorator(func: FunctionType) -> None:
                dispatch_dict[node_type] = func
            return decorator

        def map_eval_elements(elts: List[ast.AST]) -> GeneratorType:
            """Evaluate a collection of elements - used for constructing tuple, list, and set literals."""
            return map(lambda node: self.evaluate_ast(node, frame), elts)

        def evaluate_compound_statement(compound_statement: List[ast.AST]) -> None:
            """Evaluates compound statements or 'suites' of code in the terminology of the Python docs."""
            for statement in compound_statement:
                self.evaluate_ast(statement, frame)

        # AST node handlers
        @dispatch(ast.Module)
        def module():
            module_code = node.body
            evaluate_compound_statement(module_code)

        @dispatch(ast.Assign)
        def assign():
            value = self.evaluate_ast(node.value, frame)
            for x in node.targets:
                # handles tuple unpacking and multiple assignment (e.g `a, b = c, d = 3, 5`)
                # actual binding of name to value is delegated to handler of ast.Name node type
                if isinstance(x, ast.Tuple) or isinstance(x, ast.List):
                    assert len(x.elts) == len(
                        value), "unpacking assignment error"
                    for i, element in enumerate(x.elts):
                        self.evaluate_ast(element, frame, value=value[i])
                else:
                    self.evaluate_ast(x, frame, value=value)

        @dispatch(ast.Call)
        def call():
            # function evaluation procedure: evaluate function -> evaluate arguments -> apply function to arguments
            function = self.evaluate_ast(node.func, frame)
            
            args = [self.evaluate_ast(arg, frame) for arg in node.args]
            keywords = {keyword.arg: self.evaluate_ast(
                keyword.value, frame) for keyword in node.keywords}

            return function(*args, **keywords)

        @dispatch(ast.Name)
        def name():
            if isinstance(node.ctx, ast.Store):
                assert 'value' in kwargs
                frame.define(node.id, kwargs['value'])

            elif isinstance(node.ctx, ast.Load):
                return frame.get(node.id)

            elif isinstance(node.ctx, ast.Del):
                frame.delete_binding(node.id)

        @dispatch(ast.While)
        def while_form():
            # Python's exception handling machinery allows us to break out of normal control flow in host interpreter
            try:
                while self.evaluate_ast(node.test, frame):
                    try:
                        evaluate_compound_statement(node.body)
                    except ContinueSignal:
                        continue
                else:
                    evaluate_compound_statement(node.orelse)
            except BreakSignal:
                return

        @dispatch(ast.Break)
        def break_statement():
            raise BreakSignal()

        @dispatch(ast.Continue)
        def continue_statement():
            raise ContinueSignal()

        @dispatch(ast.Expr)
        def expr():
            # return value is ignored (unless we are in interactive mode): see https://docs.python.org/3.10/library/ast.html#ast.Expr
            val = self.evaluate_ast(node.value, frame)
            if self.repl:
                return val

        @dispatch(ast.Pass)
        def pass_form():
            pass

        @dispatch(ast.Constant)
        def constant():
            return node.value

        @dispatch(ast.FunctionDef)
        def function_def():
            function = AstFunction(node.body, node.args,
                                   frame, self, node.name, node.lineno)
            frame.define(node.name, function)

        @dispatch(ast.If)
        def if_form():
            predicate = self.evaluate_ast(node.test, frame)
            if predicate:
                evaluate_compound_statement(node.body)
            else:
                evaluate_compound_statement(node.orelse)

        @dispatch(ast.Compare)
        def compare():
            # https://docs.python.org/3/reference/expressions.html#comparisons
            # Handles all possible chained comparisons, including things like `x < y > z` (though why you would ever write such a thing is beyond me)
            
            values = [self.evaluate_ast(node.left, frame), self.evaluate_ast(node.comparators[0], frame)]
            if not COMPARISON_OPERATIONS[type(node.ops[0])](values[-2], values[-1]):
                return False

            for i in range(1, len(node.ops)):
                values.append(self.evaluate_ast(node.comparators[i], frame))
                if not COMPARISON_OPERATIONS[type(node.ops[i])](values[-2], values[-1]):
                    return False

            return True

        @dispatch(ast.Return)
        def return_statement():
            return_value = self.evaluate_ast(node.value, frame)
            raise ReturnSignal(return_value)

        @dispatch(ast.UnaryOp)
        def unary_op():
            return UNARY_OPERATIONS[type(node.op)](self.evaluate_ast(node.operand, frame))

        @dispatch(ast.BinOp)
        def binary_op():
            return BINARY_OPERATIONS[type(node.op)](self.evaluate_ast(node.left, frame), self.evaluate_ast(node.right, frame))

        @dispatch(ast.Tuple)
        def create_tuple():
            # case for assignment is already handled in assign function, so
            # we only have to deal with the case where a literal tuple is being created
            return tuple(map_eval_elements(node.elts))

        @dispatch(ast.List)
        def create_list():
            # same as create_tuple: don't have to worry about Store() case
            return list(map_eval_elements(node.elts))

        @dispatch(ast.Set)
        def create_set():
            return set(map_eval_elements(node.elts))

        @dispatch(ast.Dict)
        def create_dict():
            if node.keys[-1] == None:
                return dict(zip(map_eval_elements(node.keys[:-1]), map_eval_elements(node.values[:-1])),
                            **self.evaluate_ast(node.values[-1], frame))

            return dict(zip(map_eval_elements(node.keys), map_eval_elements(node.values)))

        @dispatch(ast.IfExp)
        def if_expression():
            if self.evaluate_ast(node.test, frame):
                return self.evaluate_ast(node.body, frame)
            else:
                return self.evaluate_ast(node.orelse, frame)

        @dispatch(ast.Attribute)
        def attribute():
            if isinstance(node.ctx, ast.Load):
                return getattr(self.evaluate_ast(node.value, frame), node.attr)
            elif isinstance(node.ctx, ast.Store):
                setattr(self.evaluate_ast(node.value, frame),
                        node.attr, kwargs['value'])
            elif isinstance(node.ctx, ast.Del):
                delattr(self.evaluate_ast(node.value, frame), node.attr)

        @dispatch(ast.Subscript)
        def subscript():
            raise NotImplementedError()

        @dispatch(ast.Slice)
        def create_slice():
            raise NotImplementedError()

        @dispatch(ast.AugAssign)
        def augmented_assign():
            # need to get value first, so temporarily change ctx to store then change it back in order to actually do the assignment
            node.target.ctx = ast.Load()
            left_value = self.evaluate_ast(node.target, frame)
            node.target.ctx = ast.Store()

            right_value = self.evaluate_ast(node.value, frame)
            # unfortunately because augmented assignment is a statement, it behaves slightly differently than normal binary operation
            # i.e x += y is not guaranteed to be equal to x = x + y
            # for a good example of this see lists: += is essentially equivalent to x.extend(y), which mutates the list

            if isinstance(node.op, ast.Add):
                left_value += right_value
            elif isinstance(node.op, ast.Sub):
                left_value -= right_value
            elif isinstance(node.op, ast.Mult):
                left_value *= right_value
            elif isinstance(node.op, ast.Div):
                left_value /= right_value
            elif isinstance(node.op, ast.FloorDiv):
                left_value //= right_value
            elif isinstance(node.op, ast.Mod):
                left_value %= right_value
            elif isinstance(node.op, ast.Pow):
                left_value **= right_value
            elif isinstance(node.op, ast.LShift):
                left_value <<= right_value
            elif isinstance(node.op, ast.RShift):
                left_value >>= right_value
            elif isinstance(node.op, ast.BitOr):
                left_value |= right_value
            elif isinstance(node.op, ast.BitXor):
                left_value ^= right_value
            elif isinstance(node.op, ast.BitAnd):
                left_value &= right_value
            elif isinstance(node.op, ast.MatMult):
                left_value @= right_value
            else:
                raise NotImplementedError('unrecognized augmented assignement')

            self.evaluate_ast(node.target, frame, value=left_value)

        @dispatch(ast.Assert)
        def assert_statement():
            if not self.evaluate_ast(node.test, frame):
                raise AssertionError(self.evaluate_ast(node.msg, frame))

        @dispatch(ast.Delete)
        def delete():
            raise NotImplementedError()

        @dispatch(ast.Lambda)
        def lambda_expression():
            function = AstFunction(node.body, node.args,
                                   frame, self, 'lambda', node.lineno, is_lambda=True)
            return function

        @dispatch(ast.Expression)
        def expression():
            return self.evaluate_ast(node.body, frame)

        @dispatch(ast.Interactive)
        def interactive():
            for i, statement in enumerate(node.body):
                val = self.evaluate_ast(statement, frame)
                if i == len(node.body) - 1:
                    return val

        return dispatch_dict[type(node)]()


def visualize_execution(original_eval_fn, file_contents: str):
    file_lines = file_contents.splitlines()
    def eval_with_visual(self, node: ast.AST, frame: Frame, **kwargs):
        ast_string = (
            f'Python AST (depth 2)\n'
            f'{astpretty.pformat(node, max_depth=2)}\n'
            f'{"-" * 50}'
        )
        python_string = (
            f'Python Source\n'
        )
        if isinstance(node, ast.Module) or isinstance(node, ast.Interactive):
            python_string += '<source file>'
        elif node.lineno == node.end_lineno:
            python_string += f'Line number {node.lineno}, columns {node.col_offset} to {node.end_col_offset}\n'
            python_string += file_lines[node.lineno - 1] + '\n'
            python_string += ' ' * node.col_offset + '˜' * (node.end_col_offset - node.col_offset)
        else:
            python_string += f'Line numbers {node.lineno} to {node.end_lineno}\n'
            python_string += '\n'.join(file_lines[node.lineno - 1 : node.end_lineno])
        
        python_string += '\n'
        python_string += '-' * 50

        result = original_eval_fn(node, frame, **kwargs)
        result_string = (
            f'Result\n'
            f'{result}\n'
            f'{"-" * 50}'
        )

        print(ast_string)
        print(python_string)
        print(result_string)

        return result
    return eval_with_visual

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--repl', action="store_const", const=True)
    args.add_argument('infile', nargs='?', type=argparse.FileType('r'),
                      default=sys.stdin)
    args.add_argument('--visualize', action="store_const", const=True)

    args = args.parse_args()
    
    interpreter = AstInterpreter(repl=args.repl)

    # lmao bruh
    sys.setrecursionlimit(10000)

    if args.repl:
        original_eval = interpreter.evaluate_ast
        while True:
            line = input('>>> ')
            abstract_syntax_tree = ast.parse(line, mode='single')
            if args.visualize:
                AstInterpreter.evaluate_ast = visualize_execution(original_eval, line)
            result = interpreter.evaluate_ast(abstract_syntax_tree, global_frame)
            if result is not None:
                print(result)
    else:
        file_contents = args.infile.read()
        if args.visualize:
            AstInterpreter.evaluate_ast = visualize_execution(interpreter.evaluate_ast, file_contents)

        abstract_syntax_tree = ast.parse(file_contents)
        interpreter.evaluate_ast(abstract_syntax_tree, global_frame)


if __name__ == "__main__":
    main()
