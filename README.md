# miniclosure

(Unsafe) Rust wrapper for a closure/function pointer with a single type- and lifetime-erased reference argument.

Internally uses the small function optimization, providing 56b / 60b of closure storage space on x64 / x86.

Used by [`minits`](https://github.com/alex05447/minits) and `miniecs`.