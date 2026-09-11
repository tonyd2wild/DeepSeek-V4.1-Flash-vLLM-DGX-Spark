# verify5.py: prove both SM120 kernels load from the image's FlashInfer cache without compiling at runtime.
# Runs inside overlay5 with a GPU (build_overlay5.sh). Prints "VERIFY <kernel>: HIT|MISS ..." lines.
#
# Why build_and_load() and not try_load(): in FlashInfer 0.7.0rc1 JitSpec.try_load() can return None for a module that
# is fully built and loads instantly (observed on a GB10 fleet: "VERIFY mxfp8: MISS 1.4s" while build_and_load()
# returned the module in 0.0 s and nothing compiled at serving time). is_compiled plus a timed load is the real gate.
import time


def check(name, spec_fn, load_fn=None):
    spec = spec_fn()
    ic = getattr(spec, "is_compiled", None)
    compiled = ic() if callable(ic) else ic
    t = time.time()
    try:
        (load_fn or spec.build_and_load)()
        err = None
    except Exception as e:  # loading needs a GPU; report rather than crash the build
        err = "%s: %s" % (type(e).__name__, str(e)[:120])
    dt = time.time() - t
    hit = bool(compiled) and dt < 15 and err is None
    state = "HIT" if hit else ("MISS-COMPILED" if compiled is False or dt >= 15 else "MISS")
    print("VERIFY %s: %s %.1fs (is_compiled=%s%s)" % (name, state, dt, compiled, "" if err is None else ", " + err), flush=True)


from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as g1
check("mxfp8", g1)
from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module as g2
t = time.time()
g2()
dt = time.time() - t
print("VERIFY sparse_mla: %s %.1fs" % ("HIT" if dt < 15 else "MISS-COMPILED", dt), flush=True)
