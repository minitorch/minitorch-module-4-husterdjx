from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # slope = (f(x_i + epsilon) - f(x_i - epsilon)) / (2 * epsilon)
    vals_list = list(vals)
    x_i = vals_list[arg]
    vals_list[arg] = x_i + epsilon
    f_plus = f(*vals_list)
    vals_list[arg] = x_i - epsilon
    f_minus = f(*vals_list)
    slope = (f_plus - f_minus) / (2 * epsilon)
    return slope
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):  # Protocol == interface
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:  # "" means forward declaration
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    UNVISITED = 0
    TMP_VISITED = 1
    PERM_VISITED = 2
    mark_map = {}
    L = []

    def visit(variable: Variable):
        if variable.is_constant():
            return
        if mark_map.get(variable.unique_id, UNVISITED) == PERM_VISITED:
            return
        if mark_map.get(variable.unique_id, UNVISITED) == TMP_VISITED:
            raise ValueError("Cycle detected")
        mark_map[variable.unique_id] = TMP_VISITED
        for parent in variable.parents:
            visit(parent)
        mark_map[variable.unique_id] = PERM_VISITED
        L.insert(0, variable)

    visit(variable)
    return L
    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # Call topological sort to get an ordered queue
    topo_order = topological_sort(variable)
    # Create a dictionary of Scalars and current derivatives
    deriv_map = {}
    deriv_map[variable.unique_id] = deriv
    for v in topo_order:  # topo order保证了访问到v的时候，v已经积累完成了所有的导数
        if v.is_leaf():
            v.accumulate_derivative(deriv_map[v.unique_id])
        else:  # is not leaf
            cur_deriv = deriv_map.get(v.unique_id, 0)  # important!
            for v_, d_ in v.chain_rule(cur_deriv):
                if v_.unique_id in deriv_map:
                    deriv_map[v_.unique_id] += d_
                else:
                    deriv_map[v_.unique_id] = d_
    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
