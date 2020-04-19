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

type ClosureExecutorFn = fn(&mut [u8], *const ());
type ClosureExecutorFnMut = fn(&mut [u8], *mut ());

enum ClosureExecutor {
    None,
    /// `FnMut` closure. Single immutable ref arg. Executes multiple times.
    Fn(ClosureExecutorFn),
    /// `FnOnce` closure. Single immutable ref arg. Executes once.
    FnOnce(ClosureExecutorFn),
    /// `FnMut` closure. Single mutable ref arg. Executes multiple times.
    FnMut(ClosureExecutorFnMut),
    /// `FnOnce` closure. Single mutable ref arg. Executes once.
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
/// Argument type and lifetime is erased during storage with [`new`] \ [`new_mut`] \ [`once`] \ [`once_mut`] \ [`store`] \ [`store_mut`] \ [`store_once`] \ [`store_once_mut`].
/// It's entirely up to the user to ensure the stored
/// closure is passed the correct argument type when calling [`execute`] \ [`execute_mut`] \ [`execute_once`] \ [`execute_once_mut`].
/// Stores closures with any lifetime. It is up to the caller to guarantee that any
/// borrows live until the call to [`execute`] \ [`execute_mut`] \ [`execute_once`] \ [`execute_once_mut`].
///
/// Static storage size (`CLOSURE_STORAGE_SIZE`) is determined by `CLOSURE_HOLDER_SIZE` constant and the platform function pointer size.
/// Namely, up to `CLOSURE_HOLDER_SIZE - mem::size_of<fn()>` bytes are used to store the closure in the object.
/// Closures larger than `CLOSURE_STORAGE_SIZE` are stored on the heap.
///
/// [`new`]: #method.new
/// [`new_mut`]: #method.new_mut
/// [`once`]: #method.once
/// [`once_mut`]: #method.once_mut
/// [`store`]: #method.store
/// [`store_mut`]: #method.store_mut
/// [`store_once`]: #method.store_once
/// [`store_once_mut`]: #method.store_once_mut
/// [`execute`]: #method.execute
/// [`execute_mut`]: #method.execute_mut
/// [`execute_once`]: #method.execute_once
/// [`execute_once_mut`]: #method.execute_once_mut
pub struct ClosureHolder {
    storage: ClosureStorage,
    executor: ClosureExecutor,
}

enum ClosureStorage {
    Static(MaybeUninit<StaticClosureStorage>),
    Dynamic(Option<Box<[u8]>>),
}

impl ClosureHolder {
    /// Creates an empty `ClosureHolder`.
    pub fn empty() -> Self {
        ClosureHolder {
            executor: ClosureExecutor::None,
            storage: ClosureStorage::Static(MaybeUninit::<StaticClosureStorage>::uninit()),
        }
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute`].
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
    pub unsafe fn new<'a, F, A>(f: F) -> Self
    where
        F: FnMut(&A) + 'a,
    {
        let mut result = ClosureHolder::empty();
        result.store(f);
        result
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute_mut`].
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
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn new_mut<'a, F, A>(f: F) -> Self
    where
        F: FnMut(&mut A) + 'a,
    {
        let mut result = ClosureHolder::empty();
        result.store_mut(f);
        result
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute_once`].
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once`].
    /// The caller guarantees that the following call to [`execute_once`] passes the correct argument type.
    ///
    /// [`execute_once`]: #method.execute_once
    pub unsafe fn once<'a, F, A>(f: F) -> Self
    where
        F: FnOnce(&A) + 'a,
    {
        let mut result = ClosureHolder::empty();
        result.store_once(f);
        result
    }

    /// Creates a holder which contains the closure `f` for later execution with [`execute_once_mut`].
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once_mut`].
    /// The caller guarantees that the following call to [`execute_once_mut`] passes the correct argument type.
    ///
    /// [`execute_once_mut`]: #method.execute_once_mut
    pub unsafe fn once_mut<'a, F, A>(f: F) -> Self
    where
        F: FnOnce(&mut A) + 'a,
    {
        let mut result = ClosureHolder::empty();
        result.store_once_mut(f);
        result
    }

    /// Stores the closure `f` in the holder for later execution with [`execute`].
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
    pub unsafe fn store<'a, F, A>(&mut self, f: F)
    where
        F: FnMut(&A) + 'a,
    {
        self.store_impl(
            ClosureExecutor::Fn(|storage, arg| {
                let arg: &A = &*(arg as *const _);

                let f: &mut F = &mut *(storage.as_mut_ptr() as *mut _);

                f(arg);
            }),
            f,
        );
    }

    /// Stores the closure `f` in the holder for later execution with [`execute_mut`].
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
    /// [`execute_mut`]: #method.execute_mut
    pub unsafe fn store_mut<'a, F, A>(&mut self, f: F)
    where
        F: FnMut(&mut A) + 'a,
    {
        self.store_impl(
            ClosureExecutor::FnMut(|storage, arg| {
                let arg: &mut A = &mut *(arg as *mut _);

                let f: &mut F = &mut *(storage.as_mut_ptr() as *mut _);

                f(arg);
            }),
            f,
        );
    }

    /// Stores the closure `f` in the holder for later execution with [`execute_once`].
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once`].
    /// The caller guarantees that the following call to [`execute_once`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the holder already contains a closure.
    ///
    /// [`execute_once`]: #method.execute_once
    pub unsafe fn store_once<'a, F, A>(&mut self, f: F)
    where
        F: FnOnce(&A) + 'a,
    {
        self.store_impl(
            ClosureExecutor::FnOnce(|storage, arg| {
                let arg: &A = &*(arg as *const _);

                let f = ptr::read::<F>(storage.as_mut_ptr() as *mut _);

                f(arg);
            }),
            f,
        );
    }

    /// Stores the closure `f` in the holder for later execution with [`execute_once_mut`].
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once_mut`].
    /// The caller guarantees that the following call to [`execute_once_mut`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the holder already contains a closure.
    ///
    /// [`execute_once_mut`]: #method.execute_once_mut
    pub unsafe fn store_once_mut<'a, F, A>(&mut self, f: F)
    where
        F: FnOnce(&mut A) + 'a,
    {
        self.store_impl(
            ClosureExecutor::FnOnceMut(|storage, arg| {
                let arg: &mut A = &mut *(arg as *mut _);

                let f = ptr::read::<F>(storage.as_mut_ptr() as *mut _);

                f(arg);
            }),
            f,
        );
    }

    /// If the `ClosureHolder` is not empty, returns `true`; otherwise returns `false`.
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
    pub unsafe fn try_execute<'a, A>(&mut self, arg: &'a A) -> bool {
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
    pub unsafe fn try_execute_once<'a, A>(&mut self, arg: &'a A) -> bool {
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
    pub unsafe fn try_execute_mut<'a, A>(&mut self, arg: &'a mut A) -> bool {
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
    pub unsafe fn try_execute_once_mut<'a, A>(&mut self, arg: &'a mut A) -> bool {
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
    pub unsafe fn execute<'a, A>(&mut self, arg: &'a A) {
        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::FnOnce(_) => {
                panic!("Tried to execute an `FnOnce` immutable arg closure via `execute`.")
            }
            ClosureExecutor::FnMut(_) => {
                panic!("Tried to execute an `FnMut` mutable arg closure via `execute`.")
            }
            ClosureExecutor::FnOnceMut(_) => {
                panic!("Tried to execute an `FnOnce` mutable arg closure via `execute`.")
            }
            ClosureExecutor::Fn(executor) => executor,
        };

        let arg = arg as *const _ as *const ();

        match &mut self.storage {
            ClosureStorage::Static(storage) => {
                executor(&mut *storage.as_mut_ptr(), arg);
            }
            ClosureStorage::Dynamic(storage) => {
                executor(
                    &mut *storage
                        .as_mut()
                        .expect("Tried to execute an empty closure."),
                    arg,
                );
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
    pub unsafe fn execute_once<'a, A>(&mut self, arg: &'a A) {
        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::Fn(_) => {
                panic!("Tried to execute an `FnMut` immutable arg closure via `execute_once`.")
            }
            ClosureExecutor::FnMut(_) => {
                panic!("Tried to execute an `FnMut` mutable arg closure via `execute_once`.")
            }
            ClosureExecutor::FnOnceMut(_) => {
                panic!("Tried to execute an `FnOnce` mutable arg closure via `execute_once`.")
            }
            ClosureExecutor::FnOnce(executor) => {
                let executor = executor;
                self.executor = ClosureExecutor::None;
                executor
            }
        };

        let arg = arg as *const _ as *const ();

        match &mut self.storage {
            ClosureStorage::Static(storage) => {
                executor(&mut *storage.as_mut_ptr(), arg);
            }
            ClosureStorage::Dynamic(storage) => {
                executor(
                    &mut *storage.take().expect("Tried to execute an empty closure."),
                    arg,
                );
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
    pub unsafe fn execute_mut<'a, A>(&mut self, arg: &'a mut A) {
        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::Fn(_) => {
                panic!("Tried to execute an `FnMut` immutable arg closure via `execute_mut`.")
            }
            ClosureExecutor::FnOnce(_) => {
                panic!("Tried to execute an `FnOnce` immutable arg closure via `execute_mut`.")
            }
            ClosureExecutor::FnOnceMut(_) => {
                panic!("Tried to execute an `FnOnce` mutable arg closure via `execute_mut`.")
            }
            ClosureExecutor::FnMut(executor) => executor,
        };

        let arg = arg as *mut _ as *mut ();

        match &mut self.storage {
            ClosureStorage::Static(storage) => {
                executor(&mut *storage.as_mut_ptr(), arg);
            }
            ClosureStorage::Dynamic(storage) => {
                executor(
                    &mut *storage
                        .as_mut()
                        .expect("Tried to execute an empty closure."),
                    arg,
                );
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
    pub unsafe fn execute_once_mut<'a, A>(&mut self, arg: &'a mut A) {
        let executor = match self.executor {
            ClosureExecutor::None => panic!("Tried to execute an empty closure."),
            ClosureExecutor::Fn(_) => {
                panic!("Tried to execute an `FnMut` immutable arg closure via `execute_once_mut`.")
            }
            ClosureExecutor::FnMut(_) => {
                panic!("Tried to execute an `FnMut` mutable arg closure via `execute_once_mut`.")
            }
            ClosureExecutor::FnOnce(_) => {
                panic!("Tried to execute an `FnOnce` immutable arg closure via `execute_once_mut`.")
            }
            ClosureExecutor::FnOnceMut(executor) => {
                let executor = executor;
                self.executor = ClosureExecutor::None;
                executor
            }
        };

        let arg = arg as *mut _ as *mut ();

        match &mut self.storage {
            ClosureStorage::Static(storage) => {
                executor(&mut *storage.as_mut_ptr(), arg);
            }
            ClosureStorage::Dynamic(storage) => {
                executor(
                    &mut *storage.take().expect("Tried to execute an empty closure."),
                    arg,
                );
            }
        }
    }

    unsafe fn store_closure<'a, F: Sized>(f: F) -> ClosureStorage {
        let size = mem::size_of::<F>();

        if size > CLOSURE_STORAGE_SIZE {
            let storage = vec![0u8; size].into_boxed_slice();
            let ptr = &*storage as *const _ as *mut F;

            ptr::write(ptr, f);

            ClosureStorage::Dynamic(Some(storage))
        } else {
            let mut storage = MaybeUninit::<StaticClosureStorage>::zeroed();
            let ptr = storage.as_mut_ptr() as *mut F;

            ptr::write(ptr, f);

            ClosureStorage::Static(storage)
        }
    }

    unsafe fn store_impl<F: Sized>(&mut self, e: ClosureExecutor, f: F) {
        assert!(
            self.is_none(),
            "Tried to store a closure in an occupied `ClosureHolder`."
        );

        self.executor = e;
        self.storage = ClosureHolder::store_closure(f);
    }

    #[cfg(test)]
    fn is_static(&self) -> bool {
        assert!(!self.is_none());

        if let ClosureStorage::Static(_) = &self.storage {
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    fn is_dynamic(&self) -> bool {
        assert!(!self.is_none());

        if let ClosureStorage::Dynamic(_) = &self.storage {
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let mut h = ClosureHolder::empty();
        assert!(h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_once());
        assert!(!h.is_mut());
        assert!(!h.is_once_mut());

        let arg: usize = 7;

        unsafe {
            assert!(!h.try_execute_once(&arg));
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
    fn double_store_mut() {
        let mut h = unsafe { ClosureHolder::new_mut(|_arg: &mut usize| println!("Hello")) };

        unsafe {
            h.store_mut(|_arg: &mut usize| println!(" world!"));
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
    #[should_panic(expected = "Tried to store a closure in an occupied `ClosureHolder`.")]
    fn double_store_once_mut() {
        let mut h = unsafe { ClosureHolder::once_mut(|_arg: &mut usize| println!("Hello")) };

        unsafe {
            h.store_once_mut(|_arg: &mut usize| println!(" world!"));
        }
    }

    #[test]
    fn basic() {
        // `FnMut` closure, immutable arg,
        // references mutable outer state.

        let x = 7;
        let mut y = 0;

        let mut h = unsafe {
            ClosureHolder::new(|arg: &usize| {
                assert_eq!(x, 7);
                assert_eq!(*arg, 9);

                y += x + *arg;
            })
        };

        assert!(!h.is_none());
        assert!(h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        let arg: usize = 9;

        unsafe {
            h.execute(&arg);
        }

        assert_eq!(y, x + arg);

        assert!(!h.is_none());
        assert!(h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            assert!(h.try_execute(&arg));
        }

        assert_eq!(y, x + arg + x + arg);

        assert!(!h.is_none());
        assert!(h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());
    }

    #[test]
    fn basic_mut_closure() {
        // `FnMut` closure, immutable arg,
        // references mutable outer state and has mutable inner state.

        let mut x = 0;
        let mut y = 0;

        let mut h = unsafe {
            let y = &mut y;

            ClosureHolder::new(move |arg: &usize| {
                assert_eq!(x, *arg);
                x += 1;

                assert_eq!(*y, *arg);
                *y = x;
            })
        };

        assert!(!h.is_none());
        assert!(h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            h.execute(&0usize);
        }

        assert_eq!(x, 0);
        assert_eq!(y, 1);

        assert!(!h.is_none());
        assert!(h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            h.execute(&1usize);
        }

        assert_eq!(x, 0);
        assert_eq!(y, 2);

        assert!(!h.is_none());
        assert!(h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());
    }

    #[test]
    fn basic_mut() {
        // `FnMut` closure, mutable arg,
        // references mutable outer state.

        let x = 7;
        let mut y = 0;

        let mut h = unsafe {
            ClosureHolder::new_mut(|arg: &mut usize| {
                assert_eq!(x, 7);

                y += x + *arg;

                *arg += 1;
            })
        };

        assert!(!h.is_none());
        assert!(!h.is_some());
        assert!(h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        let mut arg: usize = 9;

        unsafe {
            h.execute_mut(&mut arg);
        }

        assert_eq!(arg, 10);
        assert_eq!(y, x + arg - 1);

        assert!(!h.is_none());
        assert!(!h.is_some());
        assert!(h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            assert!(h.try_execute_mut(&mut arg));
        }

        assert_eq!(arg, 11);
        assert_eq!(y, x + arg - 2 + x + arg - 1);

        assert!(!h.is_none());
        assert!(!h.is_some());
        assert!(h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());
    }

    #[test]
    fn basic_once() {
        // `FnOnce` closure, immutable arg,
        // references mutable outer state.

        let x = 7;
        let mut y = 0;

        let mut h = unsafe {
            ClosureHolder::once(|arg: &usize| {
                assert_eq!(x, 7);
                assert_eq!(*arg, 9);

                y += x + *arg;
            })
        };

        assert!(!h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(h.is_once());
        assert!(!h.is_once_mut());

        let arg: usize = 9;

        unsafe {
            h.execute_once(&arg);
        }

        assert_eq!(y, x + arg);

        assert!(h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            assert!(!h.try_execute_once(&arg));
        }

        unsafe {
            h.store_once(|arg: &usize| {
                assert_eq!(x, 7);
                assert_eq!(*arg, 9);

                y += x + *arg;
            });
        }

        assert!(!h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            h.execute_once(&arg);
        }

        assert_eq!(y, x + arg + x + arg);

        assert!(h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            assert!(!h.try_execute_once(&arg));
        }

        assert_eq!(y, x + arg + x + arg);

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

        assert!(!g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(g.is_once());
        assert!(!g.is_once_mut());

        unsafe {
            g.execute_once(&arg);
        }

        assert!(g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(!g.is_once());
        assert!(!g.is_once_mut());

        unsafe {
            assert!(!g.try_execute_once(&arg));

            g.store_once(|_arg: &usize| {
                println!("{} {}", "Hello", 42);
            });
        }

        assert!(!g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(g.is_once());
        assert!(!g.is_once_mut());

        unsafe {
            g.execute_once(&arg);
        }

        assert!(g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(!g.is_once());
        assert!(!g.is_once_mut());

        unsafe {
            assert!(!g.try_execute_once(&arg));
        }
    }

    #[test]
    fn basic_once_mut() {
        // `FnOnce` closure, mutable arg,
        // references mutable outer state.

        let x = 7;
        let mut y = 0;

        let mut h = unsafe {
            ClosureHolder::once_mut(|arg: &mut usize| {
                assert_eq!(x, 7);
                assert_eq!(*arg, 9);

                y += x + *arg;

                *arg += 1;
            })
        };

        assert!(!h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(h.is_once_mut());

        let mut arg: usize = 9;

        unsafe {
            h.execute_once_mut(&mut arg);
        }

        assert_eq!(y, x + arg - 1);
        assert_eq!(arg, 10);

        assert!(h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            assert!(!h.try_execute_once_mut(&mut arg));
        }

        unsafe {
            h.store_once_mut(|arg: &mut usize| {
                assert_eq!(x, 7);
                assert_eq!(*arg, 10);

                y += x + *arg;

                *arg += 1;
            });
        }

        assert!(!h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(h.is_once_mut());

        unsafe {
            h.execute_once_mut(&mut arg);
        }

        assert_eq!(y, x + arg - 1 + x + arg - 2);
        assert_eq!(arg, 11);

        assert!(h.is_none());
        assert!(!h.is_some());
        assert!(!h.is_mut());
        assert!(!h.is_once());
        assert!(!h.is_once_mut());

        unsafe {
            assert!(!h.try_execute_once_mut(&mut arg));
        }

        assert_eq!(y, x + arg - 1 + x + arg - 2);
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

        assert!(!g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(!g.is_once());
        assert!(g.is_once_mut());

        unsafe {
            g.execute_once_mut(&mut arg);
        }

        assert_eq!(arg, 12);

        assert!(g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(!g.is_once());
        assert!(!g.is_once_mut());

        unsafe {
            assert!(!g.try_execute_once_mut(&mut arg));

            g.store_once_mut(|arg: &mut usize| {
                println!("{} {}", "Hello", 42);

                assert_eq!(*arg, 12);
                *arg += 1;
            });
        }

        assert!(!g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(!g.is_once());
        assert!(g.is_once_mut());

        unsafe {
            g.execute_once_mut(&mut arg);
        }

        assert_eq!(arg, 13);

        assert!(g.is_none());
        assert!(!g.is_some());
        assert!(!g.is_mut());
        assert!(!g.is_once());
        assert!(!g.is_once_mut());

        unsafe {
            assert!(!g.try_execute_once_mut(&mut arg));
        }
    }

    #[test]
    fn static_storage() {
        let capture = [0u8; CLOSURE_STORAGE_SIZE];

        let h = unsafe { ClosureHolder::once(move |_arg: &usize| println!("{}", capture.len())) };

        assert!(h.is_static());
    }

    #[test]
    fn dynamic_storage() {
        let capture = [0u8; CLOSURE_STORAGE_SIZE + 1];

        let h = unsafe { ClosureHolder::once(move |_arg: &usize| println!("{}", capture.len())) };

        assert!(h.is_dynamic());
    }

    #[test]
    #[should_panic(expected = "Tried to execute an empty closure.")]
    fn execute_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute(&0usize);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an `FnOnce` immutable arg closure via `execute`.")]
    fn execute_fn_once() {
        unsafe {
            let mut h = ClosureHolder::once(|_arg: &usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute(&0usize));
            h.execute(&0usize);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an `FnMut` mutable arg closure via `execute`.")]
    fn execute_fn_mut() {
        unsafe {
            let mut h = ClosureHolder::new_mut(|_arg: &mut usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute(&0usize));
            h.execute(&0usize);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an `FnOnce` mutable arg closure via `execute`.")]
    fn execute_fn_once_mut() {
        unsafe {
            let mut h = ClosureHolder::once_mut(|_arg: &mut usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute(&0usize));
            h.execute(&0usize);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an empty closure.")]
    fn execute_once_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute_once(&0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnMut` immutable arg closure via `execute_once`."
    )]
    fn execute_once_fn() {
        unsafe {
            let mut h = ClosureHolder::new(|_arg: &usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_once(&0usize));
            h.execute_once(&0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnMut` mutable arg closure via `execute_once`."
    )]
    fn execute_once_fn_mut() {
        unsafe {
            let mut h = ClosureHolder::new_mut(|_arg: &mut usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_once(&0usize));
            h.execute_once(&0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnOnce` mutable arg closure via `execute_once`."
    )]
    fn execute_once_fn_once_mut() {
        unsafe {
            let mut h = ClosureHolder::once_mut(|_arg: &mut usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_once(&0usize));
            h.execute_once(&0usize);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an empty closure.")]
    fn execute_mut_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnMut` immutable arg closure via `execute_mut`."
    )]
    fn execute_mut_fn() {
        unsafe {
            let mut h = ClosureHolder::new(|_arg: &usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_mut(&mut 0usize));
            h.execute_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnOnce` immutable arg closure via `execute_mut`."
    )]
    fn execute_mut_fn_once() {
        unsafe {
            let mut h = ClosureHolder::once(|_arg: &usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_mut(&mut 0usize));
            h.execute_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnOnce` mutable arg closure via `execute_mut`."
    )]
    fn execute_mut_fn_once_mut() {
        unsafe {
            let mut h = ClosureHolder::once_mut(|_arg: &mut usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_mut(&mut 0usize));
            h.execute_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(expected = "Tried to execute an empty closure.")]
    fn execute_once_mut_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute_once_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnMut` immutable arg closure via `execute_once_mut`."
    )]
    fn execute_once_mut_fn() {
        unsafe {
            let mut h = ClosureHolder::new(|_arg: &usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_once_mut(&mut 0usize));
            h.execute_once_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnMut` mutable arg closure via `execute_once_mut`."
    )]
    fn execute_once_mut_fn_mut() {
        unsafe {
            let mut h = ClosureHolder::new_mut(|_arg: &mut usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_once_mut(&mut 0usize));
            h.execute_once_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute an `FnOnce` immutable arg closure via `execute_once_mut`."
    )]
    fn execute_once_mut_fn_once() {
        unsafe {
            let mut h = ClosureHolder::once(|_arg: &usize| {
                println!("Hello.");
            });

            assert!(!h.try_execute_once_mut(&mut 0usize));
            h.execute_once_mut(&mut 0usize);
        }
    }
}
