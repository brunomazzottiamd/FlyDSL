# FlyDSL Documentation Code Snippets

Extracted from all documentation Markdown files. Snippets appear in the same order as in their source files.

## `README.md`

### Snippet 1 — `@flyc.kernel` / `@flyc.jit` API

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu

@flyc.kernel
def my_kernel(arg_a: fx.Tensor, arg_b: fx.Tensor, n: fx.Constexpr[int]):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    # ... kernel body using layout ops ...

@flyc.jit
def launch(arg_a: fx.Tensor, arg_b: fx.Tensor, n: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(arg_a, arg_b, n).launch(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )
```

```
Status : [x] pass  [ ] fail
```

---

### Snippet 2 — Hierarchical Kernel Control (tiled copy)

```python
import flydsl.expr as fx

# Define thread and value layouts for tiled copy
thr_layout = fx.make_layout((THR_M, THR_N), (1, THR_M))
val_layout = fx.make_layout((VAL_M, VAL_N), (1, VAL_M))

# Create tiled copy with vectorized atoms
copy_atom = fx.make_copy_atom(fx.CopyAtomUniversalCopyType.get(32))
layout_thr_val = fx.raked_product(thr_layout, val_layout)
tile_mn = fx.make_tile([fx.make_layout(THR_M, 1), fx.make_layout(VAL_M, 1)])
tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

# Partition tensor across blocks and threads
thr_copy = tiled_copy.get_slice(tid)
partition_src = thr_copy.partition_S(block_tile_A)
partition_dst = thr_copy.partition_D(register_fragment)

# Execute copy
fx.copy(copy_atom, partition_src, partition_dst)
```

```
Status : [ ] pass  [x] fail
Error  : NameError: name 'THR_M' is not defined
```

---

### Snippet 3 — Minimal VecAdd Example

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    # Partition tensors by block
    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))

    # Load to registers, compute, store via copy atoms
    RABTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1),
                              fx.AddressSpace.Register)
    copyAtom = fx.make_copy_atom(fx.CopyAtomUniversalCopyType.get(32))
    rA = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
    rB = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
    rC = fx.memref_alloca(RABTy, fx.make_layout(1, 1))

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

@flyc.jit
def vectorAdd(
    A: fx.Tensor, B: fx.Tensor, C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    vectorAddKernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream,
    )

# Usage
n = 128
A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
C = torch.zeros(n, dtype=torch.float32).cuda()
vectorAdd(A, B, C, n, n + 1, stream=torch.cuda.Stream())

torch.cuda.synchronize()
print("Result correct:", torch.allclose(C, A + B))
```

```
Status : [ ] pass  [x] fail
Error  : AttributeError: module 'flydsl.expr' has no attribute 'CopyAtomUniversalCopyType'
```

---

## `docs/cute_layout_algebra_guide.md`

### Snippet 1 — Core Types construction

```python
shape = fx.make_shape(128, 64)
stride = fx.make_stride(1, 128)    # Column-major
layout = fx.make_layout(shape, stride)
coord = fx.make_coord(3, 5)
```

```
Status : [ ] pass  [x] fail
Error  : NameError: name 'fx' is not definedimport flydsl.expr as fx
         After adding 'import flydsl.expr as fx':
         RuntimeError: An MLIR function requires a Context but none was provided in the call or from the surrounding environment. Either pass to the function with a 'context=' argument or establish a default using 'with Context():'
```

---

### Snippet 2 — Kernel development pattern

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    # Kernel body — use layout algebra here
    ...

@flyc.jit
def launch(
    A: fx.Tensor, B: fx.Tensor, C,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    my_kernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream,
    )
```

```
Status : [x] pass  [ ] fail
```

---

### Snippet 3 — Tensor construction with layout algebra

```python
# Create a buffer tensor from a tensor argument (AMD buffer descriptor)
A = fx.rocdl.make_buffer_tensor(A)

# Partition using layout algebra
tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
tA = fx.slice(tA, (None, bid))

# Register fragment
frag = fx.make_fragment_like(partition_src)
```

```
Status : [ ] pass  [x] fail
Error  : NameError: name 'fx' is not defined
         After adding 'import flydsl.expr as fx'
         NameError: name 'A' is not defined
```

---

### Snippet 4 — LDS allocation

```python
from flydsl.utils.smem_allocator import SmemAllocator

allocator = SmemAllocator(ctx, arch="gfx942")
lds_gen = allocator.allocate_array(T.f16(), num_elems=128*64)
allocator.finalize()

base = allocator.get_base()
lds_ptr = lds_gen(base)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 5 — XOR-based swizzle (manual implementation)

```python
# XOR-based swizzle at 16-byte granularity (manual implementation)
col_swizzled = col_bytes ^ ((row % k_blocks16) << 4)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 6 — Copy atoms and tiled copies

```python
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# Create tiled copy via raked product
thr_layout = fx.make_layout((4, 1), (1, 1))
val_layout = fx.make_layout((1, 8), (1, 1))
layout_thr_val = fx.raked_product(thr_layout, val_layout)
tile_mn = fx.make_tile(4, 8)
tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

# Get thread slice
thr_copy = tiled_copy.get_slice(tid)
src_partition = thr_copy.partition_S(src_tensor)
dst_partition = thr_copy.partition_D(dst_tensor)

# Execute copy
fx.copy(copy_atom, src_partition, dst_partition)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 7 — Buffer loads (AMD-specific)

```python
A_buf = fx.rocdl.make_buffer_tensor(A)

# Use buffer copy atoms for efficient memory access
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 8 — MFMA compute operations

```python
mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))

# Partition and execute GEMM
thr_mma = tiled_mma.thr_slice(tid)
frag_A = thr_mma.make_fragment_A(partition_A)
frag_B = thr_mma.make_fragment_B(partition_B)
frag_C = thr_mma.make_fragment_C(partition_C)
fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 9 — K64-byte micro-step pattern

```python
for ku in range(tile_k_bytes // 64):
    a_val = lds_load_pack_k32(...)   # Load A from LDS
    b_val = load_b_pack_k32(...)     # Load B from GMEM
    c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_val, b_val, c_acc)
    # second half
    a_val2 = lds_load_pack_k32(...)
    b_val2 = load_b_pack_k32(...)
    c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_val2, b_val2, c_acc)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 10 — Workgroup barrier

```python
fx.gpu.barrier()
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 11 — Compilation and execution pattern

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

# Define kernel and launch wrapper
@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, ...):
    ...

@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, C, ...,
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, C, ...).launch(
        grid=(...), block=(...), stream=stream,
    )

# Call the jit function — compilation happens automatically on first call
launch(A_torch, B_torch, C_torch, ..., stream=torch.cuda.Stream())
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 12 — Complete GEMM example with layout algebra

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

block_m, block_n, block_k = 64, 64, 8

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tileA = fx.make_tile(block_m, block_k)
    tileB = fx.make_tile(block_n, block_k)
    tileC = fx.make_tile(block_m, block_n)

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    bA = fx.slice(fx.zipped_divide(A, tileA), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, tileB), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, tileC), (None, bid))

    # MFMA atom setup
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    # Tiled copies for A, B, C
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)

    # Partition and copy to register fragments
    frag_A = thr_mma.make_fragment_A(thr_mma.partition_A(bA))
    frag_B = thr_mma.make_fragment_B(thr_mma.partition_B(bB))
    frag_C = thr_mma.make_fragment_C(thr_mma.partition_C(bC))

    # ... copy data to fragments, then GEMM ...
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    # Store result back to C
    # ...

@flyc.jit
def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
             stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/layout_system_guide.md`

### Snippet 1 — Layout construction API

```python
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T

# Shapes and strides (static constants auto-materialized)
shape = fx.make_shape(8, 16)              # !fly.int_tuple<(8, 16)>
stride = fx.make_stride(1, 8)             # !fly.int_tuple<(1, 8)>
layout = fx.make_layout(shape, stride)    # !fly.layout<(8, 16):(1, 8)>

# Shorthand — pass Python tuples directly
layout = fx.make_layout((8, 16), (1, 8))

# Coordinates
coord = fx.make_coord(i, j)

# Generic integer tuple
it = fx.make_int_tuple((4, 8, 2))

# Nested shapes
shape_nested = fx.make_shape(9, (4, 8))   # (9, (4, 8))

# Ordered layout — specify stride order (e.g., column-major vs row-major)
col_major = fx.make_ordered_layout((M, N), order=(0, 1))  # stride order: M-first
row_major = fx.make_ordered_layout((M, N), order=(1, 0))  # stride order: N-first

# Identity layout / tensor
identity = fx.make_identity_layout((M, N))
id_tensor = fx.make_identity_tensor((M, N))
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — `crd2idx`

```python
idx = fx.crd2idx(coord, layout)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 3 — `idx2crd`

```python
coord = fx.idx2crd(idx, layout)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 4 — Pure-arith helpers (`kernels/layout_utils.py`)

```python
from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

# Parses '(4,64):(64,1)' from the type and emits arith ops
flat_idx = crd2idx([row, col], layout_value)
coords = idx2crd(flat_idx, layout_value)
dim_val = layout_get(int_tuple, 0)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 5 — Query operations

```python
s = fx.size(layout)           # total elements (returns Int32 for static)
cs = fx.cosize(layout)        # codomain size (max index + 1)
shape = fx.get_shape(layout)
stride = fx.get_stride(layout)
v = fx.get(shape, 0)          # first dimension
r = fx.rank(shape)            # number of modes
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 6 — Composition

```python
composed = fx.composition(layout_a, layout_b)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 7 — Complement

```python
rest = fx.complement(tiler, target_size)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 8 — Coalesce

```python
simplified = fx.coalesce(layout)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 9 — Right inverse

```python
inv = fx.right_inverse(layout)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 10 — Recast layout

```python
# Convert layout from 16-bit to 8-bit elements
recasted = fx.recast_layout(layout, old_type_bits=16, new_type_bits=8)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 11 — Product operations

```python
result = fx.logical_product(layout, tiler)
result = fx.zipped_product(layout, tiler)
result = fx.raked_product(layout, tiler)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 12 — Divide operations

```python
result = fx.logical_divide(layout, divisor)
result = fx.zipped_divide(layout, divisor)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 13 — `select`

```python
selected = fx.select(int_tuple, indices=[0, 2])  # pick modes 0 and 2
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 14 — `group`

```python
grouped = fx.group(int_tuple, begin=1, end=3)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 15 — `append` / `prepend`

```python
extended = fx.append(base_tuple, new_elem)
extended = fx.prepend(base_tuple, new_elem)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 16 — `zip`

```python
zipped = fx.zip(shapes_a, shapes_b)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 17 — `slice`

```python
sliced = fx.slice(layout, coord)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 18 — MemRef operations

```python
# Allocate on-chip memory with layout
alloca = fx.memref_alloca(memref_type, layout)

# Load / store through layout
val = fx.memref_load(memref, indices)
fx.memref_store(value, memref, indices)

# Vector load / store
vec = fx.memref_load_vec(memref)
fx.memref_store_vec(vector, memref)

# Get layout from memref
ly = fx.get_layout(memref)

# Get iterator from memref
it = fx.get_iter(memref)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 19 — View and offset

```python
# Create a view from iterator + layout
view = fx.make_view(iterator, layout)

# Add offset to a pointer
ptr = fx.add_offset(ptr, offset)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 20 — Copy atom and tiled copy construction

```python
# Create copy atom (copy_op_type, elem_type)
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# Create MMA atom
mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))

# Build thread-value layout from thread and value layouts
tiler_mn, layout_tv = fx.make_layout_tv(thr_layout, val_layout)

# Make tiled copy from copy atom + layout + tile
tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)

# Make tiled copy matched to a TiledMma's A/B/C partitioning
tiled_copy_a = fx.make_tiled_copy_A(copy_atom, tiled_mma)
tiled_copy_b = fx.make_tiled_copy_B(copy_atom, tiled_mma)
tiled_copy_c = fx.make_tiled_copy_C(copy_atom, tiled_mma)

# Make tiled MMA from MMA atom + atom layout + optional permutation
tiled_mma = fx.make_tiled_mma(mma_atom, atom_layout)
tiled_mma = fx.make_tiled_mma(mma_atom, atom_layout, permutation)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 21 — Thread slicing and partitioning

```python
# Get a per-thread view of a tiled copy
thr_copy = tiled_copy.get_slice(tid)   # returns ThrCopy
src_part = thr_copy.partition_S(src)   # partition source tensor
dst_part = thr_copy.partition_D(dst)   # partition destination tensor
retiled  = thr_copy.retile(tensor)     # retile tensor to match copy atom

# Get a per-thread view of a tiled MMA
thr_mma = tiled_mma.get_slice(tid)     # returns ThrMma
part_a = thr_mma.partition_A(tensor_a)
part_b = thr_mma.partition_B(tensor_b)
part_c = thr_mma.partition_C(tensor_c)

# Create register fragments
frag_a = tiled_mma.make_fragment_A(part_a)
frag_b = tiled_mma.make_fragment_B(part_b)
frag_c = tiled_mma.make_fragment_C(part_c)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 22 — Copy and GEMM execution

```python
# Execute tiled copy
fx.copy(copy_atom, src_part, dst_part)

# Execute tiled copy with predicate mask (for boundary handling)
fx.copy(copy_atom, src_part, dst_part, pred=pred_tensor)

# Execute GEMM: D = A * B + C
fx.gemm(mma_atom, d, a, b, c)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 23 — Nested / hierarchical layout

```python
# Nested shape: 9 elements in first mode, (4, 8) = 32 elements in second
shape = fx.make_shape(9, (4, 8))
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 24 — IntTuple arithmetic

```python
# Element-wise operations on IntTuples
sum_it = fx.int_tuple_add(a, b)
diff_it = fx.int_tuple_sub(a, b)
prod_it = fx.int_tuple_mul(a, b)
quot_it = fx.int_tuple_div(a, b)

# Reduce to product
total = fx.int_tuple_product(int_tuple)

# Per-mode product (for nested tuples)
products = fx.int_tuple_product_each(int_tuple)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 25 — Printf debugging

```python
fx.printf("tid={} bid={} val={}", tid, bid, value)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/prebuilt_kernels_guide.md`

### Snippet 1 — Build LayerNorm module

```python
from kernels.layernorm_kernel import build_layernorm_module

executor = build_layernorm_module(M=32768, N=8192, dtype_str="bf16")
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — Build RMSNorm module

```python
from kernels.rmsnorm_kernel import build_rmsnorm_module

executor = build_rmsnorm_module(M=32768, N=8192, dtype_str="bf16")
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 3 — Build Softmax module

```python
from kernels.softmax_kernel import build_softmax_module

executor = build_softmax_module(M=32768, N=8192, dtype_str="bf16")
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 4 — Compile preshuffle GEMM

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

launch_fn = compile_preshuffle_gemm_a8(
    M=16, N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",
    lds_stage=2,
    use_cshuffle_epilog=False,
)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 5 — New API style (GEMM kernel)

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, buffer_ops, rocdl

@flyc.kernel
def gemm_kernel(arg_c: fx.Tensor, arg_a: fx.Tensor, ...):
    tid = gpu.thread_idx.x
    # ... uses fx.*, arith.*, buffer_ops.*, rocdl.* ...

@flyc.jit
def launch_fn(arg_c: fx.Tensor, ..., stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(arg_c, ...).launch(grid=..., block=..., stream=stream)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/testing_benchmarking_guide.md`

### Snippet 1 — pytest fixtures (`conftest.py`)

```python
@pytest.fixture
def ctx():
    """Fresh MLIR context per test with dialects registered."""
    # Creates Context, yields object with: ctx.context, ctx.module, ctx.location

@pytest.fixture
def module(ctx):
    """Provides ctx.module."""

@pytest.fixture
def insert_point(ctx):
    """Sets insertion point to module body."""
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — `perftest` decorator

```python
@perftest(num_iters=20, num_warmup=3, testGraph=False, num_rotate_args=0)
def my_kernel_test(Input, Output):
    # Kernel invocation
    ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 3 — `checkAllclose`

```python
checkAllclose(output, reference, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 4 — `verify_output`

```python
verify_output(c_out, c_ref, atol=1e-2, rtol=1e-2, msg='')
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 5 — `bench_gpu_us_torch`

```python
# Measure device time (torch CUDA events)
gpu_us = bench_gpu_us_torch(fn, warmup=20, iters=200)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 6 — `compile_to_hsaco`

```python
from tests.utils import compile_to_hsaco

hsaco = compile_to_hsaco(mlir_module, kernel_name="my_kernel")
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 7 — Weight utilities

```python
from tests.utils import pertoken_quant, shuffle_weight

# Per-token quantization (handles NaN/Inf)
quantized, scales = pertoken_quant(tensor, dtype=torch.float8_e4m3fnuz)

# Weight preshuffle for MFMA (layout 16x16)
shuffled = shuffle_weight(weight, layout=(16, 16))
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 8 — PyIR test pattern (no GPU)

```python
# tests/pyir/test_my_feature.py
import flydsl.expr as fx
from flydsl.expr.typing import T

def test_my_layout_op(ctx, insert_point):
    shape = fx.make_shape(4, 8)
    stride = fx.make_stride(8, 1)
    layout = fx.make_layout(shape, stride)
    result = fx.size(layout)
    ir_str = str(ctx.module)
    assert "fly.make_layout" in ir_str
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 9 — GPU kernel test pattern (new API)

```python
# tests/kernels/test_my_kernel.py
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
from tests.test_common import checkAllclose

@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    # ... kernel body ...

@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, N).launch(grid=(N // 256,), block=(256,), stream=stream)

def test_my_kernel():
    N = 1024
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.empty(N, device="cuda", dtype=torch.float32)

    launch(A, B, N)

    # Reference
    ref = A  # or some computation

    # Validate
    err = checkAllclose(B, ref, rtol=1e-2, atol=1e-2)
    assert err == 0, f"Mismatch: {err * 100:.2f}%"
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 10 — Benchmark test pattern

```python
from tests.kernels.benchmark_common import bench_gpu_us_torch

def benchmark_my_kernel():
    # Setup
    launch_fn = compile_my_kernel(...)

    def run():
        launch_fn(input_tensor, output_tensor)

    # Measure
    gpu_us = bench_gpu_us_torch(run, warmup=20, iters=200)

    # Compute metrics
    total_bytes = 2 * M * N * elem_size
    bandwidth_tbs = total_bytes / (gpu_us * 1e-6) / 1e12
    print(f"Time: {gpu_us:.1f} us, Bandwidth: {bandwidth_tbs:.2f} TB/s")
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/kernel_authoring_guide.md`

### Snippet 1 — Basic kernel pattern (`@flyc.kernel` + `@flyc.jit`)

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu

@flyc.kernel
def vec_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    idx = bid * 256 + tid
    # ... kernel body using arith/vector/buffer ops ...

@flyc.jit
def vec_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    N: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    vec_add_kernel(A, B, C, N).launch(
        grid=(N // 256,),
        block=(256,),
        stream=stream,
    )

# Usage:
import torch
A = torch.randn(1024, device="cuda", dtype=torch.float32)
B = torch.randn(1024, device="cuda", dtype=torch.float32)
C = torch.empty(1024, device="cuda", dtype=torch.float32)

vec_add(A, B, C, 1024)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — `fx.Tensor` parameter

```python
@flyc.kernel
def my_kernel(input: fx.Tensor, output: fx.Tensor):
    # input and output are Tensor wrappers around ir.Value (memref)
    ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 3 — `fx.Constexpr[T]` parameter

```python
@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int], dtype: fx.Constexpr[str]):
    for i in range_constexpr(N // 64):  # unrolled at compile time
        ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 4 — `fx.Int32` parameter

```python
@flyc.jit
def launch(data: fx.Tensor, size: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 5 — `fx.Stream` parameter

```python
@flyc.jit
def launch(data: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    my_kernel(data).launch(grid=(1,), block=(256,), stream=stream)

# Launch on specific stream:
stream = torch.cuda.Stream()
launch(data, stream=fx.Stream(stream))
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 6 — Custom argument types

```python
from flydsl.compiler import JitArgumentRegistry

@JitArgumentRegistry.register(MyCustomType, dsl_type=MyDslType)
class MyCustomAdaptor:
    def __init__(self, value: MyCustomType):
        self.value = value

    def __fly_types__(self):
        return [...]  # MLIR types for this argument

    def __fly_ptrs__(self):
        return [...]  # ctypes pointers for invocation
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 7 — Thread / block hierarchy

```python
from flydsl.expr import gpu

# Thread index within workgroup (returns Int32)
tid_x = gpu.thread_idx.x
tid_y = gpu.thread_idx.y
tid_z = gpu.thread_idx.z

# Block (workgroup) index within grid
bid_x = gpu.block_idx.x
bid_y = gpu.block_idx.y

# Block dimensions
bdim_x = gpu.block_dim.x

# Grid dimensions
gdim_x = gpu.grid_dim.x

# Low-level (returns raw ir.Value)
raw_tid = gpu.thread_id("x")
raw_bid = gpu.block_id("x")
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 8 — Arithmetic (`arith`)

```python
from flydsl.expr import arith
from flydsl.expr.typing import T

# Constants
c42 = arith.constant(42, index=True)     # index type
c3_14 = arith.constant(3.14, T.f32)      # f32 type

# Arithmetic (operator overloading via ArithValue)
result = a + b
result = a * 2
result = a // 4
result = a % 16

# Cast
idx = arith.index_cast(T.index, int_val)

# Select
result = arith.select(cond, true_val, false_val)

# Bitwise
result = arith.andi(a, b)
result = arith.xori(a, b)
result = arith.shli(a, b)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 9 — Vector operations

```python
from flydsl.expr import vector

# Build vector from elements
vec = vector.from_elements(vec_type, [a, b, c, d])

# Vector store to memref
vector.store(vec, memref, [idx])

# Extract/insert
elem = vector.extractelement(vec, idx)
vec2 = vector.insertelement(vec, elem, idx)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 10 — Buffer operations

```python
from flydsl.expr import buffer_ops

# Create buffer resource descriptor from memref
rsrc = buffer_ops.create_buffer_resource(memref_value)

# Buffer load (vectorized)
data = buffer_ops.buffer_load(rsrc, byte_offset, vec_width=4)

# Buffer store
buffer_ops.buffer_store(data, rsrc, byte_offset)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 11 — ROCm high-level helpers (`rocdl`)

```python
from flydsl.expr import rocdl

# Buffer tensor — wraps a Tensor with AMD buffer resource descriptor
A_buf = rocdl.make_buffer_tensor(A)

# MFMA MMA atom constructor — returns MmaAtomCDNA3_MFMAType
atom_type = rocdl.MFMA(m=16, n=16, k=32, elem_ty_ab=fx.Float8E4M3FNUZ)

# Buffer copy atom types
copy_op = rocdl.BufferCopy128b()   # 128-bit buffer copy
copy_op = rocdl.BufferCopy64b()    # 64-bit buffer copy
copy_op = rocdl.BufferCopy32b()    # 32-bit buffer copy
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 12 — MFMA instructions

```python
result = rocdl.mfma_f32_16x16x16f16(result_type, [a, b, acc])
result = rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, [a, b, acc])
result = rocdl.mfma_i32_16x16x32_i8(result_type, [a, b, acc])
result = rocdl.mfma_f32_16x16x16bf16_1k(result_type, [a, b, acc])   # BF16 1K variant

# GFX950 scaled MFMA (MXFP4/FP6/FP8)
result = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
    result_type, [a, b, acc, cbsz, blgp, opselA, scaleA, opselB, scaleB]
)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 13 — Instruction scheduling barriers

```python
rocdl.sched_mfma(cnt)    # wait for cnt MFMA instructions to complete
rocdl.sched_vmem(cnt)    # wait for cnt VMEM reads to complete
rocdl.sched_dsrd(cnt)    # wait for cnt DS (LDS) reads to complete
rocdl.sched_dswr(cnt)    # wait for cnt DS (LDS) writes to complete
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 14 — Low-level ROCm ops

```python
# Warp shuffle
val = rocdl.ds_bpermute(idx, src)

# Buffer load/store (raw)
data = rocdl.raw_ptr_buffer_load(rsrc, offset, soffset, aux)
rocdl.raw_ptr_buffer_store(data, rsrc, offset, soffset, aux)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 15 — GPU operations

```python
from flydsl.expr import gpu

# Barrier (workgroup synchronization)
gpu.barrier()

# Shared memory address space attribute
addrspace = gpu.smem_space()
addrspace_int = gpu.smem_space(int=True)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 16 — Control flow (Python loops → MLIR SCF)

```python
@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int]):
    # Compile-time unrolled loop
    for i in range_constexpr(N):
        # This loop is fully unrolled in the generated IR
        ...

    # Runtime loop (lowered to scf.for)
    for i in range(runtime_value):
        ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 17 — `const_expr()`

```python
from flydsl.expr import const_expr

@flyc.kernel
def my_kernel(data: fx.Tensor, N: fx.Constexpr[int]):
    tile_size = const_expr(N // 4)
    for i in range_constexpr(tile_size):
        ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 18 — `SmemAllocator` (shared memory / LDS)

```python
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.expr.typing import T

# Create allocator for target architecture
allocator = SmemAllocator(None, arch="gfx942", global_sym_name="smem0")

# Allocate typed arrays
lds_a = allocator.allocate_array(T.f16, 8192)
lds_b = allocator.allocate_array(T.f16, 8192)

# Inside kernel: get base pointer and typed views
lds_base = allocator.get_base()
lds_a_ptr = lds_a(lds_base)  # SmemPtr
lds_b_ptr = lds_b(lds_base)  # SmemPtr

# Load/store through SmemPtr
val = lds_a_ptr.load([idx])
lds_b_ptr.store(val, [idx])
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 19 — Finalizing LDS allocation

```python
comp_ctx = CompilationContext.get_current()
with ir.InsertionPoint(comp_ctx.gpu_module_body):
    allocator.finalize()
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 20 — Launch configuration

```python
@flyc.jit
def launch(data: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    my_kernel(data).launch(
        grid=(num_blocks_x, num_blocks_y, num_blocks_z),
        block=(threads_x, threads_y, threads_z),
        smem=shared_mem_bytes,     # dynamic shared memory
        stream=stream,             # CUDA/HIP stream
    )
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 21 — Dynamic grid/block dimensions

```python
@flyc.jit
def launch(data: fx.Tensor, M: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    grid_x = M // 256
    my_kernel(data, M).launch(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 22 — Workgroup synchronization (barrier)

```python
from flydsl.expr import gpu

# Workgroup barrier (s_barrier)
gpu.barrier()
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 23 — Complete example: Preshuffle GEMM

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl, range_constexpr
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator

def compile_preshuffle_gemm_a8(*, M, N, K, tile_m, tile_n, tile_k,
                                 in_dtype="fp8", lds_stage=2, ...):
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    lds_a = allocator.allocate_array(T.i8, tile_m * tile_k)
    # ... more allocations ...

    @flyc.kernel
    def gemm_kernel(
        arg_c: fx.Tensor, arg_a: fx.Tensor, arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor, arg_scale_b: fx.Tensor,
        m_in: fx.Int32, n_in: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        # ... complex GEMM implementation using MFMA, LDS, tiling ...

    @flyc.jit
    def launch_fn(
        arg_c: fx.Tensor, arg_a: fx.Tensor, arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor, arg_scale_b: fx.Tensor,
        M_val: fx.Int32, N_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gemm_kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b,
                    M_val, N_val).launch(
            grid=(grid_x, grid_y), block=(256,),
            smem=smem_bytes, stream=stream,
        )

    return launch_fn
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/architecture_guide.md`

### Snippet 1 — `@flyc.jit` decorator

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.jit
def launch(a: fx.Tensor, b: fx.Tensor, n: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(a, b, n).launch(grid=(n // 256,), block=(256,), stream=stream)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — `@flyc.kernel` decorator

```python
@flyc.kernel
def my_kernel(a: fx.Tensor, b: fx.Tensor, n: fx.Constexpr[int]):
    tid = fx.gpu.thread_id("x")
    bid = fx.gpu.block_id("x")
    # ... kernel body ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 3 — `KernelLauncher`

```python
launcher = my_kernel(a, b, 1024)
launcher.launch(
    grid=(num_blocks, 1, 1),
    block=(256, 1, 1),
    smem=shared_mem_bytes,
    stream=stream_value,
)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 4 — `DslType` / `JitArgument` protocols

```python
# DslType protocol — for values used inside kernel/jit functions
class DslType(Protocol):
    @classmethod
    def __fly_construct__(cls, values: List[ir.Value]) -> "DslType": ...
    def __fly_values__(self) -> List[ir.Value]: ...

# JitArgument protocol — for values passed at the host boundary
class JitArgument(Protocol):
    def __fly_types__(self) -> List[ir.Type]: ...
    def __fly_ptrs__(self) -> List[ctypes.c_void_p]: ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 5 — Registering custom argument types

```python
from flydsl.compiler import JitArgumentRegistry

@JitArgumentRegistry.register(MyPythonType, dsl_type=MyDslType)
class MyJitArg:
    def __fly_types__(self): ...
    def __fly_ptrs__(self): ...
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/quickstart.rst`

### Snippet 1 — Minimal Vector Add kernel

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    # Partition tensors by block using layout algebra
    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))

    # Allocate register fragments, load, compute, store
    RABTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1),
                              fx.AddressSpace.Register)
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
    rB = fx.memref_alloca(RABTy, fx.make_layout(1, 1))
    rC = fx.memref_alloca(RABTy, fx.make_layout(1, 1))

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))

@flyc.jit
def vectorAdd(
    A: fx.Tensor, B: fx.Tensor, C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    vectorAddKernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream,
    )

# Usage
n = 128
A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
C = torch.zeros(n, dtype=torch.float32).cuda()
vectorAdd(A, B, C, n, n + 1, stream=torch.cuda.Stream())
torch.cuda.synchronize()
print("Result correct:", torch.allclose(C, A + B))
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/tutorials/basic_usage.rst`

### Snippet 1 — Layout creation inside a kernel

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(data: fx.Tensor):
    # Create a 2D layout: 8 rows x 16 columns, column-major
    layout = fx.make_layout((8, 16), (1, 8))
    # Index = dot(Coord, Stride) = i*1 + j*8
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — Kernel definition with `@flyc.kernel`

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(
    data: fx.Tensor,
    n: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    # ... kernel body using layout ops
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 3 — Launching a kernel with `@flyc.jit`

```python
@flyc.jit
def launch(
    data: fx.Tensor,
    n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    my_kernel(data, n).launch(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )

# Usage with PyTorch
import torch
data = torch.randn(1024, dtype=torch.float32).cuda()
launch(data, 1024, stream=torch.cuda.Stream())
torch.cuda.synchronize()
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/tutorials/kernel_development.rst`

### Snippet 1 — Tiled copies

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def copy_kernel(src: fx.Tensor, dst: fx.Tensor):
    tid = fx.thread_idx.x

    # Define thread and value layouts
    thr_layout = fx.make_layout((4, 1), (1, 1))
    val_layout = fx.make_layout((1, 8), (1, 1))

    # Create a copy atom (e.g., 128-bit buffer copy)
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

    # Build the tiled copy descriptor via raked product
    layout_thr_val = fx.raked_product(thr_layout, val_layout)
    tile_mn = fx.make_tile(4, 8)
    tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

    # Partition a tensor for this thread
    thr_copy = tiled_copy.get_slice(tid)
    partition_src = thr_copy.partition_S(block_tile)
    partition_dst = thr_copy.partition_D(fragment)

    # Execute copy
    fx.copy(copy_atom, partition_src, partition_dst)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — MFMA instructions

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def mfma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x

    # Create an MFMA atom (16x16x4 FP32)
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))

    # Partition A, B, C for this thread
    thr_mma = tiled_mma.thr_slice(tid)
    frag_A = thr_mma.make_fragment_A(partition_A)
    frag_B = thr_mma.make_fragment_B(partition_B)
    frag_C = thr_mma.make_fragment_C(partition_C)

    # Execute GEMM
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/api/compiler.rst`

### Snippet 1 — `@flyc.kernel` and `@flyc.jit`

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, n: fx.Constexpr[int]):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    # ... kernel body using layout ops ...

@flyc.jit
def launch(A: fx.Tensor, B: fx.Tensor, n: fx.Constexpr[int],
           stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, n).launch(
        grid=(grid_x, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — `flyc.from_dlpack`

```python
import flydsl.compiler as flyc

tA = flyc.from_dlpack(torch_tensor).mark_layout_dynamic(
    leading_dim=0, divisibility=4
)
launch(tA, B, n, stream=torch.cuda.Stream())
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

## `docs/api/dsl.rst`

### Snippet 1 — Import `flydsl.expr`

```python
import flydsl.expr as fx
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 2 — Arithmetic operations (`arith`)

```python
from flydsl.expr import arith

c = arith.constant(42, index=True)
v = arith.index_cast(T.index, val)
r = arith.select(cond, a, b)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 3 — Buffer operations (`buffer_ops`)

```python
from flydsl.expr import buffer_ops

rsrc = buffer_ops.create_buffer_resource(tensor, max_size=True)
data = buffer_ops.buffer_load(rsrc, offset, vec_width=4, dtype=T.i32)
buffer_ops.buffer_store(data, rsrc, offset, mask=is_valid)
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```

---

### Snippet 4 — Import `flydsl.compiler`

```python
import flydsl.compiler as flyc
```

```
Status : [ ] pass  [ ] fail
Error  : (none)
```
