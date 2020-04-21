# miniclosure

(Unsafe) wrapper for a closure/function pointer with a single type- and lifetime-erased reference argument.

Internally uses the small function optimization, providing 47b / 55b of closure storage space on x64 / x86.

Used by `minits` and `miniecs`.