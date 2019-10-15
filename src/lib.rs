#[macro_use]
extern crate static_assertions;

use std::mem::{self, MaybeUninit};
use std::ptr;

/// Total amount of memory used by the [`StaticClosureHolder`]
/// x86: 4b (executor function pointer) + 60b (closure storage) = 64b
/// x64: 8b (executor function pointer) + 56b (closure storage) = 64b
/// [`StaticClosureHolder`]: #struct.StaticClosureHolder
const CLOSURE_HOLDER_SIZE: usize = 64;
const CLOSURE_PTR_SIZE: usize = mem::size_of::<fn()>();
const CLOSURE_STORAGE_SIZE: usize = CLOSURE_HOLDER_SIZE - CLOSURE_PTR_SIZE;

type StaticClosureStorage = [u8; CLOSURE_STORAGE_SIZE];
type ClosureExecutorFn = fn(&[u8], *const ());

enum ClosureExecutor {
    None,
    FnOnce(ClosureExecutorFn),
    Fn(ClosureExecutorFn),
}

impl ClosureExecutor {
    fn is_some(&self) -> bool {
        match self {
            ClosureExecutor::None => false,
            _ => true,
        }
    }

    fn is_fn(&self) -> bool {
        match self {
            ClosureExecutor::Fn(_) => true,
            _ => false,
        }
    }
}

/// (Unsafe) wrapper for a closure/function pointer with a single type- and lifetime-erased reference argument.
///
/// Internally uses the small function optimization, providing 56b / 60b of closure storage space on x64 / x86.
///
/// # Safety
///
/// Argument type and lifetime is erased during storage with [`new`] \ [`store`]. It's entirely up to the user to ensure the stored
/// closure is passed the correct argument type when calling [`execute`].
/// Stores closures with any lifetime with [`new`] \ [`store`]. It is up to the caller to guarantee that any
/// borrows live until the call to [`execute`].
///
/// Static storage size (`CLOSURE_STORAGE_SIZE`) is determined by `CLOSURE_HOLDER_SIZE` constant and the platform function pointer size.
/// Namely, up to `CLOSURE_HOLDER_SIZE - mem::size_of<fn()>` bytes are used to store the closure in the object.
/// Closures larger than `CLOSURE_STORAGE_SIZE` are stored on the heap.
///
/// [`new`]: #method.new
/// [`store`]: #method.store
/// [`execute`]: #method.execute
pub struct ClosureHolder {
    storage: ClosureStorage,
    executor: ClosureExecutor,
}

enum ClosureStorage {
    Static(MaybeUninit<StaticClosureStorage>),
    Dynamic(Option<Box<[u8]>>),
}

assert_eq_size!(closure_storage; StaticClosureStorage, [u8; CLOSURE_STORAGE_SIZE]);

impl ClosureHolder {
    /// Creates an empty holder.
    pub fn empty() -> Self {
        ClosureHolder {
            executor: ClosureExecutor::None,
            storage: ClosureStorage::Static(MaybeUninit::<StaticClosureStorage>::uninit()),
        }
    }

    /// Creates a holder which contains the closure `f` for later execution.
    ///
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`].
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// [`execute`]: #method.execute
    pub unsafe fn new<'any, F, ARG>(f: F) -> Self
    where
        F: FnMut(&ARG) + 'any,
    {
        let mut result = ClosureHolder::empty();
        result.store(f);
        result
    }

    /// Creates a holder which contains the closure `f` for later execution.
    ///
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`].
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// [`execute`]: #method.execute
    pub unsafe fn once<'any, F, ARG>(f: F) -> Self
    where
        F: FnOnce(&ARG) + 'any,
    {
        let mut result = ClosureHolder::empty();
        result.store_once(f);
        result
    }

    /// Stores the closure `f` in the holder for later execution.
    ///
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`].
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the holder already contains a closure.
    ///
    /// [`execute`]: #method.execute
    pub unsafe fn store_once<'any, F, ARG>(&mut self, f: F)
    where
        F: FnOnce(&ARG) + 'any,
    {
        self.store_impl(f, true);
    }

    /// Stores the closure `f` in the holder for later execution.
    ///
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`].
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the holder already contains a closure.
    ///
    /// [`execute`]: #method.execute
    pub unsafe fn store<'any, F, ARG>(&mut self, f: F)
    where
        F: FnMut(&ARG) + 'any,
    {
        self.store_impl(f, false);
    }

    pub unsafe fn store_impl<'any, F, ARG>(&mut self, f: F, once: bool)
    where
        F: FnOnce(&ARG) + 'any,
    {
        assert!(!self.is_some(), "Tried to store a closure in an occupied `ClosureHolder`.");

        let size = mem::size_of::<F>();

        if size > CLOSURE_STORAGE_SIZE {
            let storage = vec![0u8; size].into_boxed_slice();
            let ptr = &*storage as *const _ as *mut F;

            ptr::write(ptr, f);

            self.storage = ClosureStorage::Dynamic(Some(storage));
        } else {
            let mut storage = MaybeUninit::<StaticClosureStorage>::uninit();
            let ptr = storage.as_mut_ptr() as *mut F;

            ptr::write(ptr, f);

            self.storage = ClosureStorage::Static(storage);
        }

        let executor = |storage: &[u8], arg: *const ()| {
            let arg: &ARG = mem::transmute(arg);

            let f = ptr::read::<F>(storage.as_ptr() as *const F);

            f(arg);
        };

        self.executor = if once {
            ClosureExecutor::FnOnce(executor)
        } else {
            ClosureExecutor::Fn(executor)
        };
    }

    /// If the `ClosureHolder` is not empty, returns `true`; otherwise returns `false`.
    pub fn is_some(&self) -> bool {
        self.executor.is_some()
    }

    /// If the `ClosureHolder` is not empty, executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// If the closure was stored via [`once`] \ [`store_once`], the `ClosureHolder` becomes empty after the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`once`] \ [`store`] \ [`store_once`].
    ///
    /// [`new`]: #method.new
    /// [`once`]: #method.once
    /// [`store`]: #method.store
    /// [`store_once`]: #method.store_once
    pub unsafe fn try_execute_once<'any, ARG>(&mut self, arg: &'any ARG) -> bool {
        if self.is_some() {
            self.execute_once(arg);
            true
        } else {
            false
        }
    }

    /// If the `ClosureHolder` is not empty and was stored via [`new`] \ [`store`], executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`store`].
    ///
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn try_execute<'any, ARG>(&self, arg: &'any ARG) -> bool {
        if self.executor.is_fn() {
            self.execute(arg);
            true
        } else {
            false
        }
    }

    /// Executes the stored closure unconditionally.
    ///
    /// If the closure was stored via [`once`] \ [`store_once`], the `ClosureHolder` becomes empty after the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`once`] \ [`store`] \ [`store_once`].
    ///
    /// # Panics
    ///
    /// Panics if the `ClosureHolder` is empty.
    ///
    /// [`new`]: #method.new
    /// [`once`]: #method.once
    /// [`store`]: #method.store
    /// [`store_once`]: #method.store_once
    pub unsafe fn execute_once<'any, ARG>(&mut self, arg: &'any ARG) {
        let arg: *const () = mem::transmute(arg);

        let (executor, once) = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::FnOnce(executor) => {
                let executor = executor;
                self.executor = ClosureExecutor::None;
                (executor, true)
            },
            ClosureExecutor::Fn(executor) => (executor, false),
        };

        match &mut self.storage {
            ClosureStorage::Static(storage) => {
                let storage = &*storage.as_ptr();
                executor(storage, arg);
            },
            ClosureStorage::Dynamic(storage) => {
                if once {
                    let storage = storage.take().expect("Tried to execute an empty closure.");
                    let storage = &*storage;
                    executor(storage, arg);
                } else {
                    let storage = &*storage.as_ref().expect("Tried to execute an empty closure.");
                    executor(storage, arg);
                }
            },
        }
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`new`] \ [`store`].
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`store`].
    ///
    /// # Panics
    ///
    /// Panics if the `ClosureHolder` is empty.
    /// Panics if the closure was stored via [`once`] \ [`store_once`].
    ///
    /// [`new`]: #method.new
    /// [`once`]: #method.once
    /// [`store`]: #method.store
    /// [`store_once`]: #method.store_once
    pub unsafe fn execute<'any, ARG>(&self, arg: &'any ARG) {
        let arg: *const () = mem::transmute(arg);

        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::FnOnce(_) => panic!("Tried to execute an `FnOnce` closure via `execute`."),
            ClosureExecutor::Fn(executor) => executor,
        };

        match &self.storage {
            ClosureStorage::Static(storage) => {
                let storage = &*storage.as_ptr();
                executor(storage, arg);
            },
            ClosureStorage::Dynamic(storage) => {
                let storage = &*storage.as_ref().expect("Tried to execute an empty closure.");
                executor(storage, arg);
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let mut h = ClosureHolder::empty();
        let arg = 7usize;

        assert!(!h.is_some());

        unsafe {
            assert!(!h.try_execute_once(&arg));
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an empty closure.")]
    fn empty_execute() {
        let mut h = ClosureHolder::empty();
        let arg = 7usize;

        assert!(!h.is_some());

        unsafe {
            assert!(!h.try_execute_once(&arg));
            h.execute_once(&arg);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to store a closure in an occupied `ClosureHolder`.")]
    fn double_store() {
        let mut h = unsafe {
            ClosureHolder::once(|_arg: &usize| println!("Hello"))
        };

        unsafe {
            h.store_once(|_arg: &usize| println!(" world!"));
        }
    }

    #[test]
    fn basic_once() {
        // Move closure.
        let x = 7;
        let mut y = 0;

        let mut h = unsafe {
            ClosureHolder::once(move |arg: &usize| {
                assert_eq!(x, 7);
                assert_eq!(y, 0);
                assert_eq!(*arg, 9);

                y = x + *arg;
            })
        };

        let arg = 9usize;

        assert!(h.is_some());

        unsafe {
            h.execute_once(&arg);
        }

        assert_eq!(y, 0);

        assert!(!h.is_some());

        unsafe {
            assert!(!h.try_execute_once(&arg));
        }

        unsafe {
            h.store_once(move |arg: &usize| {
                assert_eq!(x, 7);
                assert_eq!(y, 0);
                assert_eq!(*arg, 9);

                y = x + *arg;
            });
        }

        assert!(h.is_some());

        unsafe {
            h.execute_once(&arg);
        }

        assert!(!h.is_some());

        unsafe {
            assert!(!h.try_execute_once(&arg));
        }

        assert_eq!(x, 7);
        assert_eq!(y, 0);

        // Static closure.
        static HELLO: &'static str = "Hello";
        static FORTY_TWO: usize = 42;

        let mut g = unsafe {
            ClosureHolder::once(|_arg: &usize| {
                assert_eq!(HELLO, "Hello");
                assert_eq!(FORTY_TWO, 42);

                println!("{} {}", HELLO, FORTY_TWO);
            })
        };

        assert!(g.is_some());

        unsafe {
            g.execute_once(&arg);
        }

        assert!(!g.is_some());

        unsafe {
            assert!(!g.try_execute_once(&arg));

            g.store_once(|_arg: &usize| {
                println!("{} {}", "Hello", 42);
            });
        }

        assert!(g.is_some());

        unsafe {
            g.execute_once(&arg);
        }

        assert!(!g.is_some());

        unsafe {
            assert!(!g.try_execute_once(&arg));
        }
    }

    #[test]
    fn basic() {
        let x = 7;
        let mut y = 0;

        let h = unsafe {
            ClosureHolder::new(|arg: &usize| {
                assert_eq!(x, 7);
                assert_eq!(*arg, 9);

                y = y + x + *arg;
            })
        };

        let arg = 9usize;

        assert!(h.is_some());

        unsafe {
            h.execute(&arg);
        }

        assert_eq!(y, 7 + 9);

        assert!(h.is_some());

        unsafe {
            assert!(h.try_execute(&arg));
        }

        assert_eq!(y, 7 + 9 + 7 + 9);
    }

    #[test]
    fn storage_overflow() {
        let large_capture = [0u8; CLOSURE_HOLDER_SIZE];

        let mut h =
            unsafe { ClosureHolder::once(move |_arg: &usize| println!("{}", large_capture.len())) };

        let arg = 7usize;

        unsafe {
            h.execute_once(&arg);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an `FnOnce` closure via `execute`.")]
    fn execute_once() {
        // Move closure.
        let x = 7;
        let mut y = 0;

        let h = unsafe {
            ClosureHolder::once(move |arg: &usize| {
                assert_eq!(x, 7);
                assert_eq!(y, 0);
                assert_eq!(*arg, 9);

                y = x + *arg;
            })
        };

        let arg = 9usize;

        assert!(h.is_some());

        unsafe {
            h.execute(&arg);
        }
    }
}
