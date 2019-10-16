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
type ClosureExecutorFnMut = fn(&[u8], *mut ());

enum ClosureExecutor {
    None,

    // Single immutable ref arg. Executes multiple times.
    Fn(ClosureExecutorFn),

    // Single immutable ref arg. Executes once.
    FnOnce(ClosureExecutorFn),

    // Single immutable ref arg. Executes multiple times.
    FnMut(ClosureExecutorFnMut),

    // Single mutable ref arg. Executes once.
    FnOnceMut(ClosureExecutorFnMut),
}

impl ClosureExecutor {
    fn is_none(&self) -> bool {
        match self {
            ClosureExecutor::None => true,
            _ => false,
        }
    }

    fn is_some(&self) -> bool {
        match self {
            ClosureExecutor::Fn(_) => true,
            _ => false,
        }
    }

    fn is_once(&self) -> bool {
        match self {
            ClosureExecutor::FnOnce(_) => true,
            _ => false,
        }
    }

    fn is_mut(&self) -> bool {
        match self {
            ClosureExecutor::FnMut(_) => true,
            _ => false,
        }
    }

    fn is_once_mut(&self) -> bool {
        match self {
            ClosureExecutor::FnOnceMut(_) => true,
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
/// Argument type and lifetime is erased during storage with [`new`] \ [`new_mut`] \ [`store`] \ [`store_mut`].
/// It's entirely up to the user to ensure the stored
/// closure is passed the correct argument type when calling [`execute`] \ [`execute_mut`].
/// Stores closures with any lifetime. It is up to the caller to guarantee that any
/// borrows live until the call to [`execute`] \ [`execute_mut`].
///
/// Static storage size (`CLOSURE_STORAGE_SIZE`) is determined by `CLOSURE_HOLDER_SIZE` constant and the platform function pointer size.
/// Namely, up to `CLOSURE_HOLDER_SIZE - mem::size_of<fn()>` bytes are used to store the closure in the object.
/// Closures larger than `CLOSURE_STORAGE_SIZE` are stored on the heap.
///
/// [`new`]: #method.new
/// [`new_mut`]: #method.new_mut
/// [`store`]: #method.store
/// [`store_mut`]: #method.store_mut
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
    /// Creates an empty `ClosureHolder`.
    pub fn empty() -> Self {
        ClosureHolder {
            executor: ClosureExecutor::None,
            storage: ClosureStorage::Static(MaybeUninit::<StaticClosureStorage>::uninit()),
        }
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute`] (not [`execute_mut`]).
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`].
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// [`execute`]: #method.execute
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn new<'any, F, ARG>(f: F) -> Self
    where
        F: FnMut(&ARG) + 'any,
    {
        let mut result = ClosureHolder::empty();
        result.store(f);
        result
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute_mut`] (not [`execute`]).
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_mut`].
    /// The caller guarantees that the following call to [`execute_mut`] passes the correct argument type.
    ///
    /// [`execute`]: #method.execute
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn new_mut<'any, F, ARG>(f: F) -> Self
    where
        F: FnMut(&mut ARG) + 'any,
    {
        let mut result = ClosureHolder::empty();
        result.store_mut(f);
        result
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute`] (not [`execute_mut`]).
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`].
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// [`execute`]: #method.execute
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn once<'any, F, ARG>(f: F) -> Self
    where
        F: FnOnce(&ARG) + 'any,
    {
        let mut result = ClosureHolder::empty();
        result.store_once(f);
        result
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute_mut`] (not [`execute`]).
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_mut`].
    /// The caller guarantees that the following call to [`execute_mut`] passes the correct argument type.
    ///
    /// [`execute`]: #method.execute
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn once_mut<'any, F, ARG>(f: F) -> Self
    where
        F: FnOnce(&mut ARG) + 'any,
    {
        let mut result = ClosureHolder::empty();
        result.store_once_mut(f);
        result
    }

    /// Stores the closure `f` in the holder for later execution with [`execute`] (not [`execute_mut`]).
    ///
    /// Closure takes a single immutable reference argument.
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
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn store<'any, F, ARG>(&mut self, f: F)
    where
        F: FnMut(&ARG) + 'any,
    {
        self.store_impl(f, false);
    }

    /// Stores the closure `f` in the holder for later execution with [`execute_mut`] (not [`execute`]).
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_mut`].
    /// The caller guarantees that the following call to [`execute_mut`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the holder already contains a closure.
    ///
    /// [`execute`]: #method.execute
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn store_mut<'any, F, ARG>(&mut self, f: F)
    where
        F: FnMut(&mut ARG) + 'any,
    {
        self.store_mut_impl(f, false);
    }

    /// Stores the closure `f` in the holder for later execution with [`execute`] (not [`execute_mut`]).
    ///
    /// Closure takes a single immutable reference argument.
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
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn store_once<'any, F, ARG>(&mut self, f: F)
    where
        F: FnOnce(&ARG) + 'any,
    {
        self.store_impl(f, true);
    }

    /// Stores the closure `f` in the holder for later execution with [`execute_mut`] (not [`execute`]).
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_mut`].
    /// The caller guarantees that the following call to [`execute_mut`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the holder already contains a closure.
    ///
    /// [`execute`]: #method.execute
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn store_once_mut<'any, F, ARG>(&mut self, f: F)
    where
        F: FnOnce(&mut ARG) + 'any,
    {
        self.store_mut_impl(f, true);
    }

    pub fn is_none(&self) -> bool {
        self.executor.is_none()
    }

    /// If the `ClosureHolder` is not empty and was stored via [`new`] \ [`store`],
    /// returns `true` (the user may call [`execute`]); otherwise returns `false`.
    ///
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    /// [`execute`]: #method.execute
    pub fn is_some(&self) -> bool {
        self.executor.is_some()
    }

    /// If the `ClosureHolder` is not empty and was stored via [`once`] \ [`store_once`],
    /// returns `true` (the user may call [`execute_once`]); otherwise returns `false`.
    ///
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    /// [`execute_once`]: #method.execute_once
    pub fn is_once(&self) -> bool {
        self.executor.is_once()
    }

    /// If the `ClosureHolder` is not empty and was stored via [`new_mut`] \ [`store_mut`],
    /// returns `true` (the user may call [`execute_mut`]); otherwise returns `false`.
    ///
    /// [`new_mut`]: #method.new
    /// [`store_mut`]: #method.store
    /// [`execute_mut`]: #method.execute_mut
    pub fn is_mut(&self) -> bool {
        self.executor.is_mut()
    }

    /// If the `ClosureHolder` is not empty and was stored via [`once_mut`] \ [`store_once_mut`],
    /// returns `true` (the user may call [`execute_once_mut`]); otherwise returns `false`.
    ///
    /// [`once_mut`]: #method.new
    /// [`store_once_mut`]: #method.store
    /// [`execute_once_mut`]: #method.execute_once_mut
    pub fn is_once_mut(&self) -> bool {
        self.executor.is_once_mut()
    }

    /// If the `ClosureHolder` is not empty and was stored via [`new`] \ [`store`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`store`].
    ///
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn try_execute<'any, ARG>(&self, arg: &'any ARG) -> bool {
        if self.executor.is_some() {
            self.execute(arg);
            true
        } else {
            false
        }
    }

    /// If the `ClosureHolder` is not empty and was stored via [`once`] \ [`store_once`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// `ClosureHolder` becomes empty if the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`once`] \ [`store`] \ [`store_once`].
    ///
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    pub unsafe fn try_execute_once<'any, ARG>(&mut self, arg: &'any ARG) -> bool {
        if self.is_once() {
            self.execute_once(arg);
            true
        } else {
            false
        }
    }

    /// If the `ClosureHolder` is not empty and was stored via [`new_mut`] \ [`store_mut`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new_mut`] \ [`store_mut`].
    ///
    /// [`new_mut`]: #method.new_mut
    /// [`store_mut`]: #method.store_mut
    pub unsafe fn try_execute_mut<'any, ARG>(&self, arg: &'any mut ARG) -> bool {
        if self.executor.is_mut() {
            self.execute_mut(arg);
            true
        } else {
            false
        }
    }

    /// If the `ClosureHolder` is not empty and was stored via [`once_mut`] \ [`store_once_mut`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// `ClosureHolder` becomes empty if the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once_mut`] \ [`store_once_mut`].
    ///
    /// [`once_mut`]: #method.once_mut
    /// [`store_once_mut`]: #method.store_once_mut
    pub unsafe fn try_execute_once_mut<'any, ARG>(&mut self, arg: &'any mut ARG) -> bool {
        if self.is_once_mut() {
            self.execute_once_mut(arg);
            true
        } else {
            false
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
    /// Panics if the closure was not stored via [`new`] \ [`store`].
    ///
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn execute<'any, ARG>(&self, arg: &'any ARG) {
        let arg: *const () = mem::transmute(arg);

        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::FnOnce(_) => {
                panic!("Tried to execute an `FnOnce` closure via `execute`.")
            }
            ClosureExecutor::FnMut(_) => {
                panic!("Tried to execute a mutable closure via `execute`.")
            }
            ClosureExecutor::FnOnceMut(_) => {
                panic!("Tried to execute an `FnOnce mutable closure via `execute`.")
            }
            ClosureExecutor::Fn(executor) => executor,
        };

        match &self.storage {
            ClosureStorage::Static(storage) => {
                let storage = &*storage.as_ptr();
                executor(storage, arg);
            }
            ClosureStorage::Dynamic(storage) => {
                let storage = &*storage
                    .as_ref()
                    .expect("Tried to execute an empty closure.");
                executor(storage, arg);
            }
        }
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once`] \ [`store_once`].
    ///
    /// The `ClosureHolder` becomes empty after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once`] \ [`store_once`].
    ///
    /// # Panics
    ///
    /// Panics if the `ClosureHolder` is empty.
    /// Panics if the closure was not stored via [`once`] \ [`store_once`].
    ///
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    pub unsafe fn execute_once<'any, ARG>(&mut self, arg: &'any ARG) {
        let arg: *const () = mem::transmute(arg);

        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::Fn(_) => {
                panic!("Tried to execute a non-`FnOnce` closure via `execute_once`.")
            }
            ClosureExecutor::FnMut(_) => {
                panic!("Tried to execute a non-`FnOnce` mutable closure via `execute_once`.")
            }
            ClosureExecutor::FnOnceMut(_) => {
                panic!("Tried to execute a mutable closure via `execute_once`.")
            }
            ClosureExecutor::FnOnce(executor) => {
                let executor = executor;
                self.executor = ClosureExecutor::None;
                executor
            }
        };

        match &mut self.storage {
            ClosureStorage::Static(storage) => {
                let storage = &*storage.as_ptr();
                executor(storage, arg);
            }
            ClosureStorage::Dynamic(storage) => {
                let storage = storage.take().expect("Tried to execute an empty closure.");
                let storage = &*storage;
                executor(storage, arg);
            }
        }
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`new_mut`] \ [`store_mut`].
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new_mut`] \ [`store_mut`].
    ///
    /// # Panics
    ///
    /// Panics if the `ClosureHolder` is empty.
    /// Panics if the closure was not stored via [`new_mut`] \ [`store_mut`].
    ///
    /// [`new_mut`]: #method.new_mut
    /// [`store_mut`]: #method.store_mut
    pub unsafe fn execute_mut<'any, ARG>(&self, arg: &'any mut ARG) {
        let arg: *mut () = mem::transmute(arg);

        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::Fn(_) => {
                panic!("Tried to execute an immutable closure via `execute_mut`.")
            }
            ClosureExecutor::FnOnce(_) => {
                panic!("Tried to execute an `FnOnce` immutable closure via `execute_mut`.")
            }
            ClosureExecutor::FnOnceMut(_) => {
                panic!("Tried to execute an `FnOnce` mutable closure via `execute_mut`.")
            }
            ClosureExecutor::FnMut(executor) => executor,
        };

        match &self.storage {
            ClosureStorage::Static(storage) => {
                let storage = &*storage.as_ptr();
                executor(storage, arg);
            }
            ClosureStorage::Dynamic(storage) => {
                let storage = &*storage
                    .as_ref()
                    .expect("Tried to execute an empty closure.");
                executor(storage, arg);
            }
        }
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// The `ClosureHolder` becomes empty after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once_mut`] \ [`store_once_mut`].
    ///
    /// # Panics
    ///
    /// Panics if the `ClosureHolder` is empty.
    /// Panics if the closure was not stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// [`once_mut`]: #method.once_mut
    /// [`store_once_mut`]: #method.store_once_mut
    pub unsafe fn execute_once_mut<'any, ARG>(&mut self, arg: &'any mut ARG) {
        let arg: *mut () = mem::transmute(arg);

        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::Fn(_) => {
                panic!("Tried to execute a non-`FnOnce` immutable closure via `execute_once_mut`.")
            }
            ClosureExecutor::FnMut(_) => {
                panic!("Tried to execute a non-`FnOnce` mutable closure via `execute_once_mut`.")
            }
            ClosureExecutor::FnOnce(_) => {
                panic!("Tried to execute an immutable closure via `execute_once_mut`.")
            }
            ClosureExecutor::FnOnceMut(executor) => {
                let executor = executor;
                self.executor = ClosureExecutor::None;
                executor
            }
        };

        match &mut self.storage {
            ClosureStorage::Static(storage) => {
                let storage = &*storage.as_ptr();
                executor(storage, arg);
            }
            ClosureStorage::Dynamic(storage) => {
                let storage = storage.take().expect("Tried to execute an empty closure.");
                let storage = &*storage;
                executor(storage, arg);
            }
        }
    }

    unsafe fn store_impl<'any, F, ARG>(&mut self, f: F, once: bool)
    where
        F: FnOnce(&ARG) + 'any,
    {
        assert!(
            self.is_none(),
            "Tried to store a closure in an occupied `ClosureHolder`."
        );

        self.storage = ClosureHolder::store_closure(f);

        self.executor = ClosureHolder::store_executor::<'any, F, ARG>(once);
    }

    unsafe fn store_mut_impl<'any, F, ARG>(&mut self, f: F, once: bool)
    where
        F: FnOnce(&mut ARG) + 'any,
    {
        assert!(
            self.is_none(),
            "Tried to store a closure in an occupied `ClosureHolder`."
        );

        self.storage = ClosureHolder::store_closure(f);

        self.executor = ClosureHolder::store_executor_mut::<'any, F, ARG>(once);
    }

    unsafe fn store_closure<'any, F, ARG>(f: F) -> ClosureStorage
    where
        F: FnOnce(ARG) + 'any,
    {
        let size = mem::size_of::<F>();

        if size > CLOSURE_STORAGE_SIZE {
            let storage = vec![0u8; size].into_boxed_slice();
            let ptr = &*storage as *const _ as *mut F;

            ptr::write(ptr, f);

            ClosureStorage::Dynamic(Some(storage))
        } else {
            let mut storage = MaybeUninit::<StaticClosureStorage>::uninit();
            let ptr = storage.as_mut_ptr() as *mut F;

            ptr::write(ptr, f);

            ClosureStorage::Static(storage)
        }
    }

    unsafe fn store_executor<'any, F, ARG>(once: bool) -> ClosureExecutor
    where
        F: FnOnce(&ARG) + 'any,
    {
        let executor = |storage: &[u8], arg: *const ()| {
            let arg: &ARG = mem::transmute(arg);

            let f = ptr::read::<F>(storage.as_ptr() as *const F);

            f(arg);
        };

        if once {
            ClosureExecutor::FnOnce(executor)
        } else {
            ClosureExecutor::Fn(executor)
        }
    }

    unsafe fn store_executor_mut<'any, F, ARG>(once: bool) -> ClosureExecutor
    where
        F: FnOnce(&mut ARG) + 'any,
    {
        let executor = |storage: &[u8], arg: *mut ()| {
            let arg: &mut ARG = mem::transmute(arg);

            let f = ptr::read::<F>(storage.as_ptr() as *const F);

            f(arg);
        };

        if once {
            ClosureExecutor::FnOnceMut(executor)
        } else {
            ClosureExecutor::FnMut(executor)
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
        let mut h = unsafe { ClosureHolder::new(|_arg: &usize| println!("Hello")) };

        unsafe {
            h.store(|_arg: &usize| println!(" world!"));
        }
    }

    #[test]
    #[should_panic(expected = "Tried to store a closure in an occupied `ClosureHolder`.")]
    fn double_store_once() {
        let mut h = unsafe { ClosureHolder::once(|_arg: &usize| println!("Hello")) };

        unsafe {
            h.store_once(|_arg: &usize| println!(" world!"));
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
    fn basic_mut() {
        let x = 7;
        let mut y = 0;

        let h = unsafe {
            ClosureHolder::new_mut(|arg: &mut usize| {
                assert_eq!(x, 7);

                y = y + x + *arg;

                *arg += 1;
            })
        };

        let mut arg = 9usize;

        assert!(h.is_mut());

        unsafe {
            h.execute_mut(&mut arg);
        }

        assert_eq!(y, 7 + 9);
        assert_eq!(arg, 10);

        assert!(h.is_mut());

        unsafe {
            assert!(h.try_execute_mut(&mut arg));
        }

        assert_eq!(y, 7 + 9 + 7 + 10);
        assert_eq!(arg, 11);
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

        assert!(h.is_once());

        unsafe {
            h.execute_once(&arg);
        }

        assert_eq!(y, 0);

        assert!(!h.is_once());

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

        assert!(h.is_once());

        unsafe {
            h.execute_once(&arg);
        }

        assert!(!h.is_once());

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

        assert!(g.is_once());

        unsafe {
            g.execute_once(&arg);
        }

        assert!(!g.is_once());

        unsafe {
            assert!(!g.try_execute_once(&arg));

            g.store_once(|_arg: &usize| {
                println!("{} {}", "Hello", 42);
            });
        }

        assert!(g.is_once());

        unsafe {
            g.execute_once(&arg);
        }

        assert!(!g.is_once());

        unsafe {
            assert!(!g.try_execute_once(&arg));
        }
    }

    #[test]
    fn basic_once_mut() {
        // Move closure.
        let x = 7;
        let mut y = 0;

        let mut h = unsafe {
            ClosureHolder::once_mut(move |arg: &mut usize| {
                assert_eq!(x, 7);
                assert_eq!(y, 0);
                assert_eq!(*arg, 9);

                y = x + *arg;

                *arg += 1;
            })
        };

        let mut arg = 9usize;

        assert!(h.is_once_mut());

        unsafe {
            h.execute_once_mut(&mut arg);
        }

        assert_eq!(y, 0);
        assert_eq!(arg, 10);

        assert!(h.is_none());

        unsafe {
            assert!(!h.try_execute_once_mut(&mut arg));
        }

        assert_eq!(arg, 10);

        unsafe {
            h.store_once_mut(move |arg: &mut usize| {
                assert_eq!(x, 7);
                assert_eq!(y, 0);
                assert_eq!(*arg, 10);

                y = x + *arg;

                *arg += 1;
            });
        }

        assert!(h.is_once_mut());

        unsafe {
            h.execute_once_mut(&mut arg);
        }

        assert!(h.is_none());
        assert_eq!(arg, 11);

        unsafe {
            assert!(!h.try_execute_once_mut(&mut arg));
        }

        assert_eq!(x, 7);
        assert_eq!(y, 0);
        assert_eq!(arg, 11);

        // Static closure.
        static HELLO: &'static str = "Hello";
        static FORTY_TWO: usize = 42;

        let mut g = unsafe {
            ClosureHolder::once_mut(|arg: &mut usize| {
                assert_eq!(HELLO, "Hello");
                assert_eq!(FORTY_TWO, 42);

                println!("{} {}", HELLO, FORTY_TWO);

                assert_eq!(*arg, 11);
                *arg += 1;
            })
        };

        assert!(g.is_once_mut());

        unsafe {
            g.execute_once_mut(&mut arg);
        }

        assert!(g.is_none());
        assert_eq!(arg, 12);

        unsafe {
            assert!(!g.try_execute_once_mut(&mut arg));

            g.store_once_mut(|arg: &mut usize| {
                println!("{} {}", "Hello", 42);

                assert_eq!(*arg, 12);
                *arg += 1;
            });
        }

        assert!(g.is_once_mut());

        unsafe {
            g.execute_once_mut(&mut arg);
        }

        assert!(g.is_none());

        unsafe {
            assert!(!g.try_execute_once_mut(&mut arg));
        }
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
    fn execute_fn_once() {
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

        assert!(h.is_once());

        unsafe {
            h.execute(&arg);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute a non-`FnOnce` closure via `execute_once`.")]
    fn execute_once_fn() {
        // Move closure.
        let x = 7;
        let mut y = 0;

        let mut h = unsafe {
            ClosureHolder::new(move |arg: &usize| {
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
    }
}
