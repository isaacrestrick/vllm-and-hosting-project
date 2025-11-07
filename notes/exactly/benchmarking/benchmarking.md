Exactly — that’s a clean way to think about it.

**Below (B_{sat}):**

* The GPU still has unused compute units and memory channels.
* So adding tokens (bigger batch) just fills idle capacity — *“free parallelization.”*
* The only real cost is a bit more memory traffic (copying activations / inputs in and out).
* Step time stays almost flat because the GPU can overlap those loads with computation.

**Above (B_{sat}):**

* All compute units and memory bandwidth are fully used.
* More tokens mean strictly more FLOPs to execute — the GPU can’t parallelize them further.
* Step time now grows roughly linearly with batch size.

So yes — **below (B_{sat})**, you get parallel speedup “for free,” limited only by memory bandwidth.
**Beyond (B_{sat})**, you leave the memory-bound zone and become compute-bound.
