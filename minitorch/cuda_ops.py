from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            # print("a_shape: ", a.shape, "out_shape: ", out_shape, "out_size: ", out_a.size, "blockNum: ", blockspergrid)
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = fn(in_storage[index_to_position(in_index, in_strides)])
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= out_size:
            return
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = fn(
            a_storage[index_to_position(a_index, a_strides)],
            b_storage[index_to_position(b_index, b_strides)],
        )
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32
    # print("gridDim: {}-{}-{}, blockDim: {}-{}-{}".format(cuda.gridDim.x, cuda.gridDim.y, cuda.gridDim.z, cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z))
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    if i >= size:
        return
    cache[pos] = a[i]
    cuda.syncthreads()
    index = 1
    while index < BLOCK_DIM:
        if pos % (2 * index) == 0 and pos + index < BLOCK_DIM and i + index < size:
            cache[pos] += cache[pos + index]
        index *= 2
        cuda.syncthreads()
    if pos == 0:
        out[cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x] = cache[pos]
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    print(out)
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # if out_pos >= out_size:
        #     return
        # reduce_dim_size = a_shape[reduce_dim]
        # to_index(out_pos, out_shape, out_index)
        # to_index(out_pos, out_shape, a_index)
        # a_index[reduce_dim] = pos
        # if pos < reduce_dim_size:
        #     cache[pos] = a_storage[index_to_position(a_index, a_strides)]
        # else:
        #     cache[pos] = 0.0
        # cuda.syncthreads()
        # naive
        # if pos == 0:
        #     for i in range(1, reduce_dim_size):
        #         a_index[reduce_dim] = i
        #         cache[0] = fn(
        #             a_storage[index_to_position(a_index, a_strides)], cache[0]
        #         )
        #     out[out_pos] = cache[0]

        # parallel
        # index = 1
        # while index < BLOCK_DIM:
        #     if pos % (2 * index) == 0 and pos + index < reduce_dim_size:
        #         a_index[reduce_dim] = pos + index
        #         cache[pos] = fn(a_storage[index_to_position(a_index, a_strides)],
        #                         cache[pos])
        #     index *= 2
        #     cuda.syncthreads()
        # if pos == 0:
        #     out[out_pos] = cache[0]
        # TODO: Implement for Task 3.3.

        # other
        if i < len(out):
            to_index(i, out_shape, out_index)
            out_index[reduce_dim] = 0
            out_pos = index_to_position(out_index, out_strides)
            cache[pos] = reduce_value
            for j in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = j
                a_pos = index_to_position(out_index, a_strides)
                cache[pos] = fn(cache[pos], a_storage[a_pos])
            out[out_pos] = cache[pos]
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    x = numba.cuda.blockIdx.x * numba.cuda.blockDim.x + numba.cuda.threadIdx.x
    y = numba.cuda.blockIdx.y * numba.cuda.blockDim.y + numba.cuda.threadIdx.y
    cache_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    cache_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    if x >= size or y >= size:
        return
    cache_a[x, y] = a[x * size + y]
    cache_b[x, y] = b[x * size + y]
    numba.cuda.syncthreads()
    tmp = 0.0
    for i in range(size):
        tmp += cache_a[x, i] * cache_b[i, y]
    out[x * size + y] = tmp
    # raise NotImplementedError("Need to implement for Task 3.3")


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # @https://github.com/minitorch/minitorch-module-3-zmvictor/blob/master/minitorch/cuda_ops.py
    M, N, K = out_shape[1], out_shape[2], a_shape[-1]
    acc = 0.0  # accumulator
    # 2. To calculate out[i, j], we need a[i, ...] and b[..., j]
    # Thus, each thread needs to copy one row of a and one column of b
    # 对于每个block来说，最终的结果就是一个out matrix里面block大小的结果。不同的block计算不同位置的结果，铺满整个out matrix
    for start in range(0, K, BLOCK_DIM):  # start is the starting index of the block
        # build guards to make sure we don't copy values out of bounds
        a_k = start + pj
        # copy a[i, start + pj] to a_shared[pi, pj]
        if i < M and a_k < K:
            # thread在外部循环时，每次fetch大矩阵的一个值过来。想象在a矩阵中block按一行的顺序往右移动，
            # 在b矩阵中block按一列的顺序往下移动，每次移动一个block大小，直到移动到矩阵的边界
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + a_k * a_strides[2]
            ]
        b_k = start + pi
        # copy b[start + pi, j] to b_shared[pi, pj]
        if b_k < K and j < N:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + b_k * b_strides[1] + j * b_strides[2]
            ]
        # synchronize the threads
        cuda.syncthreads()
        # 3. After copying, calculate dot product of the two blocks and add to acc
        # build a guard to make sure we don't use values out of bounds
        for k in range(BLOCK_DIM):
            if start + k < K:
                acc += a_shared[pi, k] * b_shared[k, pj]
        # 每个thread计算出一个block大小的矩阵的一行、一列乘积和后，每次外部循环就是拓展该行、该列在原矩阵的位置，
        # 最终能够计算出大矩阵的一行、一列的乘积和
    # 4. Copy acc to out[i, j]
    # Note: the number of threads is not necessarily equal to the number of elements in out
    # we need to use guard to make sure we don't copy values out of bounds
    if i < M and j < N:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc
    # raise NotImplementedError("Need to implement for Task 3.4")


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
