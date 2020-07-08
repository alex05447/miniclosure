use {
    static_assertions::{assert_eq_size, const_assert_eq},
    std::{
        mem::{self, size_of},
        ptr,
    },
};

/// Static function which
/// 1) unpacks the correct closure type from the closure holder storage,
/// 2) casts the argument ptr to the expected argument type,
/// 3) executes the closure,
/// 4) cleans up the closure holder, if necessary.
///
/// Gets passed a mutable reference to the [`ClosureHolder`]
/// and the pointer to type-erased closure argument.
///
/// It's up to the user to ensure the argument is of the type the closure expects.
///
/// [`ClosureHolder`]: struct.ClosureHolder.html
type Executor = fn(&mut ClosureHolder, *mut ());

/// Static function which
/// 1) unpacks the correct closure type from the closure holder storage,
/// 2) drops it.
///
/// Gets passed a mutable reference to the [`ClosureHolder`].
///
/// Gets called when the [`ClosureHolder`] is [`cleared`] or dropped.
///
/// [`ClosureHolder`]: struct.ClosureHolder.html
/// [`cleared`]: struct.ClosureHolder.html#method.clear
type DropHandler = fn(&mut ClosureHolder);

/// Vtable struct for a concrete closure type
/// which knows
/// 1) how to unpack/execute/cleanup the closure (via `execute`);
/// 2) how to clear / drop it (via `drop`);
/// 3) the closure's type w.r.t. its mutability / its argument mutability (via `executor_tag`);
/// 4) whether the closure storage is dynamic (boxed) (via `storage_tag`).
/// A static pointer to this is stored in the closure holder
#[repr(C)]
#[derive(Clone, Copy)]
struct ClosureVTable {
    executor_tag: fn() -> ExecutorTag,
    execute: Executor,
    drop: DropHandler,

    #[cfg(test)]
    storage_tag: fn() -> StorageTag,
}

fn default_vtable() -> &'static ClosureVTable {
    &ClosureVTable {
        executor_tag: || ExecutorTag::None,
        execute: |_, _| {},
        drop: |_| {},

        #[cfg(test)]
        storage_tag: || StorageTag::Static,
    }
}

fn storage_tag<F>() -> StorageTag {
    if size_of::<F>() > STORAGE_SIZE {
        StorageTag::Dynamic
    } else {
        StorageTag::Static
    }
}

/// Total amount of memory used by the [`ClosureHolder`]
///
/// [`ClosureHolder`]: #struct.ClosureHolder.html
const HOLDER_SIZE: usize = 64;

const VTABLE_SIZE: usize = size_of::<&'static ClosureVTable>();

/// Amount of memory left for closure capture storage left in the [`ClosureHolder`].
///
/// x86: 4b (vtable ptr) + 60b (closure storage) = 64b
/// x64: 8b (vtable ptr) + 56b (closure storage) = 64b
///
/// [`ClosureHolder`]: #struct.ClosureHolder.html
const STORAGE_SIZE: usize = HOLDER_SIZE /* 64b */
    - VTABLE_SIZE; /* 4b / 8b -> 60b / 56b */

/// Raw byte buffer used to store the closure capture, if it fits in `STORAGE_SIZE` bytes.
#[repr(C)]
#[derive(Clone, Copy)] // `Copy` needed for union storage.
struct StaticStorage([u8; STORAGE_SIZE]);

const_assert_eq!(size_of::<StaticStorage>(), STORAGE_SIZE);

/// Heap-allocated buffer used to store the closure capture, if it does not fit in `STORAGE_SIZE` bytes.
#[repr(C)]
#[derive(Clone, Copy)] // `Copy` needed for union storage.
struct DynamicStorage(*mut [u8]);

const_assert_eq!(size_of::<DynamicStorage>(), size_of::<usize>() * 2); // Fat pointer.

/// Closure capture storage.
/// Static or dynamic.
/// Tag/discriminant is encoded separately (via the vtable).
#[repr(C)]
union StorageUnion {
    _static: StaticStorage,
    _dynamic: DynamicStorage,
}

impl Default for StorageUnion {
    fn default() -> Self {
        Self {
            _static: unsafe { StaticStorage(mem::zeroed()) },
        }
    }
}

assert_eq_size!(StorageUnion, StaticStorage);

/// Explicit storage union tag, encoded separately (via the vtable).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StorageTag {
    Static = 0,
    Dynamic,
}

/// Closure executor type, encoded separately (via the vtable).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExecutorTag {
    None = 0,
    /// `FnMut` closure. Single immutable ref arg. Executes multiple times.
    Fn,
    /// `FnOnce` closure. Single immutable ref arg. Executes once.
    FnOnce,
    /// `FnMut` closure. Single mutable ref arg. Executes multiple times.
    FnMut,
    /// `FnOnce` closure. Single mutable ref arg. Executes once.
    FnOnceMut,
}

impl ExecutorTag {
    fn is_none(&self) -> bool {
        match self {
            ExecutorTag::None => true,
            _ => false,
        }
    }

    fn is_some(&self) -> bool {
        match self {
            ExecutorTag::Fn => true,
            _ => false,
        }
    }

    fn is_once(&self) -> bool {
        match self {
            ExecutorTag::FnOnce => true,
            _ => false,
        }
    }

    fn is_mut(&self) -> bool {
        match self {
            ExecutorTag::FnMut => true,
            _ => false,
        }
    }

    fn is_once_mut(&self) -> bool {
        match self {
            ExecutorTag::FnOnceMut => true,
            _ => false,
        }
    }
}

/// (Unsafe) wrapper for a closure/function pointer with a single type- and lifetime-erased reference argument.
///
/// Internally uses the small function optimization, providing 47b / 55b of closure storage space on x64 / x86.
///
/// # Safety
///
/// Argument type and lifetime is erased during storage with [`new`] \ [`new_mut`] \ [`once`] \ [`once_mut`] \ [`store`] \ [`store_mut`] \ [`store_once`] \ [`store_once_mut`].
/// It's entirely up to the user to ensure the stored
/// closure is passed the correct argument type when calling [`execute`] \ [`execute_mut`] \ [`execute_once`] \ [`execute_once_mut`].
/// Stores closures with any lifetime. It is up to the caller to guarantee that any
/// borrows live until the call to [`execute`] \ [`execute_mut`] \ [`execute_once`] \ [`execute_once_mut`],
/// or until the closure holder is [`cleared`] \ `Drop`'ped.
///
/// Static storage size (`STORAGE_SIZE`) is determined by `HOLDER_SIZE` constant and the platform function pointer size.
/// Closures larger than `STORAGE_SIZE` are stored on the heap.
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
/// [`cleared`]: #method.clear
#[repr(C)]
pub struct ClosureHolder {
    vt: &'static ClosureVTable, // offs 0b          size 4b / 8b
    storage: StorageUnion,      // offs 4b / 8b     size 60b / 56b
}

const_assert_eq!(size_of::<ClosureHolder>(), HOLDER_SIZE);

impl ClosureHolder {
    /// Creates an empty [`ClosureHolder`].
    ///
    /// [`ClosureHolder`]: #struct.ClosureHolder.html
    pub fn empty() -> Self {
        ClosureHolder {
            vt: default_vtable(),
            storage: Default::default(),
        }
    }

    /// Creates a [`holder`] which contains the closure `f` for later execution with [`execute`].
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`], [`clear`], or `Drop`.
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute`]: #method.execute
    /// [`clear`]: #method.clear
    pub unsafe fn new<F, A>(f: F) -> Self
    where
        F: FnMut(&A),
    {
        let mut result = Self::empty();
        result.store(f);
        result
    }

    /// Creates a [`holder`] which contains the closure `f` for later execution with [`execute_mut`].
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_mut`], [`clear`], or `Drop`.
    /// The caller guarantees that the following call to [`execute_mut`] passes the correct argument type.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute_mut`]: #method.execute_mut
    /// [`clear`]: #method.clear
    pub unsafe fn new_mut<F, A>(f: F) -> Self
    where
        F: FnMut(&mut A),
    {
        let mut result = Self::empty();
        result.store_mut(f);
        result
    }

    /// Creates a [`holder`] which contains the closure `f` for later execution with [`execute_once`].
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once`], [`clear`] or `Drop`.
    /// The caller guarantees that the following call to [`execute_once`] passes the correct argument type.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute_once`]: #method.execute_once
    /// [`clear`]: #method.clear
    pub unsafe fn once<F, A>(f: F) -> Self
    where
        F: FnOnce(&A),
    {
        let mut result = Self::empty();
        result.store_once(f);
        result
    }

    /// Creates a [`holder`] which contains the closure `f` for later execution with [`execute_once_mut`].
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once_mut`], [`clear`] or `Drop`.
    /// The caller guarantees that the following call to [`execute_once_mut`] passes the correct argument type.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute_once_mut`]: #method.execute_once_mut
    /// [`clear`]: #method.clear
    pub unsafe fn once_mut<F, A>(f: F) -> Self
    where
        F: FnOnce(&mut A),
    {
        let mut result = Self::empty();
        result.store_once_mut(f);
        result
    }

    /// Stores the closure `f` in the [`holder`] for later execution with [`execute`].
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute`], [`clear`], or `Drop`.
    /// The caller guarantees that the following call to [`execute`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the [`holder`] already contains a closure.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute`]: #method.execute
    /// [`clear`]: #method.clear
    pub unsafe fn store<F, A>(&mut self, f: F)
    where
        F: FnMut(&A),
    {
        let vt = &ClosureVTable {
            executor_tag: || ExecutorTag::Fn,
            execute: |h, arg| {
                (Self::closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage)))(Self::arg(
                    arg,
                ));
            },
            drop: |h| {
                Self::take_closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage)); // Closure dropped here.
                h.clear_impl(storage_tag::<F>());
            },

            #[cfg(test)]
            storage_tag: storage_tag::<F>,
        };

        self.store_impl(vt, f);
    }

    /// Stores the closure `f` in the [`holder`] for later execution with [`execute_mut`].
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may be executed multiple times.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_mut`], [`clear`], or `Drop`.
    /// The caller guarantees that the following call to [`execute_mut`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the [`holder`] already contains a closure.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute_mut`]: #method.execute_mut
    /// [`clear`]: #method.clear
    pub unsafe fn store_mut<F, A>(&mut self, f: F)
    where
        F: FnMut(&mut A),
    {
        let vt = &ClosureVTable {
            executor_tag: || ExecutorTag::FnMut,
            execute: |h, arg| {
                (Self::closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage)))(
                    Self::arg_mut(arg),
                );
            },
            drop: |h| {
                Self::take_closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage)); // Closure dropped here.
                h.clear_impl(storage_tag::<F>());
            },

            #[cfg(test)]
            storage_tag: storage_tag::<F>,
        };

        self.store_impl(vt, f);
    }

    /// Stores the closure `f` in the [`holder`] for later execution with [`execute_once`].
    ///
    /// Closure takes a single immutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once`], [`clear`] or `Drop`.
    /// The caller guarantees that the following call to [`execute_once`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the [`holder`] already contains a closure.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute_once`]: #method.execute_once
    /// [`clear`]: #method.clear
    pub unsafe fn store_once<F, A>(&mut self, f: F)
    where
        F: FnOnce(&A),
    {
        let vt = &ClosureVTable {
            executor_tag: || ExecutorTag::FnOnce,
            execute: |h, arg| {
                let f = Self::take_closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage));
                h.clear_impl(storage_tag::<F>());
                f(Self::arg(arg)); // Closure dropped here.
            },
            drop: |h| {
                Self::take_closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage)); // Closure dropped here.
                h.clear_impl(storage_tag::<F>());
            },

            #[cfg(test)]
            storage_tag: storage_tag::<F>,
        };

        self.store_impl(vt, f);
    }

    /// Stores the closure `f` in the [`holder`] for later execution with [`execute_once_mut`].
    ///
    /// Closure takes a single mutable reference argument.
    /// Stored closure may only be executed once.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the closure
    /// does not outlive its borrows, if any, until the following call to [`execute_once_mut`], [`clear`] or `Drop`.
    /// The caller guarantees that the following call to [`execute_once_mut`] passes the correct argument type.
    ///
    /// # Panics
    ///
    /// Panics if the [`holder`] already contains a closure.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`execute_once_mut`]: #method.execute_once_mut
    /// [`clear`]: #method.clear
    pub unsafe fn store_once_mut<F, A>(&mut self, f: F)
    where
        F: FnOnce(&mut A),
    {
        let vt = &ClosureVTable {
            executor_tag: || ExecutorTag::FnOnceMut,
            execute: |h, arg| {
                let f = Self::take_closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage));
                h.clear_impl(storage_tag::<F>());
                f(Self::arg_mut(arg)); // Closure dropped here.
            },
            drop: |h| {
                Self::take_closure::<F>(Self::storage(storage_tag::<F>(), &mut h.storage)); // Closure dropped here.
                h.clear_impl(storage_tag::<F>());
            },

            #[cfg(test)]
            storage_tag: storage_tag::<F>,
        };

        self.store_impl(vt, f);
    }

    /// If the [`holder`] is not empty, returns `true`; otherwise returns `false`.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    pub fn is_none(&self) -> bool {
        Self::executor_tag(self.vt).is_none()
    }

    /// If the [`holder`] is not empty and was stored via [`new`] \ [`store`],
    /// returns `true` (the user may call [`execute`]); otherwise returns `false`.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    /// [`execute`]: #method.execute
    pub fn is_some(&self) -> bool {
        Self::executor_tag(self.vt).is_some()
    }

    /// If the [`holder`] is not empty and was stored via [`once`] \ [`store_once`],
    /// returns `true` (the user may call [`execute_once`]); otherwise returns `false`.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    /// [`execute_once`]: #method.execute_once
    pub fn is_once(&self) -> bool {
        Self::executor_tag(self.vt).is_once()
    }

    /// If the [`holder`] is not empty and was stored via [`new_mut`] \ [`store_mut`],
    /// returns `true` (the user may call [`execute_mut`]); otherwise returns `false`.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new_mut`]: #method.new
    /// [`store_mut`]: #method.store
    /// [`execute_mut`]: #method.execute_mut
    pub fn is_mut(&self) -> bool {
        Self::executor_tag(self.vt).is_mut()
    }

    /// If the [`holder`] is not empty and was stored via [`once_mut`] \ [`store_once_mut`],
    /// returns `true` (the user may call [`execute_once_mut`]); otherwise returns `false`.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once_mut`]: #method.new
    /// [`store_once_mut`]: #method.store
    /// [`execute_once_mut`]: #method.execute_once_mut
    pub fn is_once_mut(&self) -> bool {
        Self::executor_tag(self.vt).is_once_mut()
    }

    /// If the [`holder`] is not empty and was stored via [`new`] \ [`store`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`store`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn try_execute<A>(&mut self, arg: &A) -> bool {
        if self.is_some() {
            self.execute(arg);
            true
        } else {
            false
        }
    }

    /// If the [`holder`] is not empty and was stored via [`once`] \ [`store_once`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// The [`holder`] becomes empty / [`cleared`] if the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`once`] \ [`store`] \ [`store_once`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    /// [`cleared`]: #method.clear
    pub unsafe fn try_execute_once<A>(&mut self, arg: &A) -> bool {
        if self.is_once() {
            self.execute_once(arg);
            true
        } else {
            false
        }
    }

    /// If the [`holder`] is not empty and was stored via [`new_mut`] \ [`store_mut`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new_mut`] \ [`store_mut`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new_mut`]: #method.new_mut
    /// [`store_mut`]: #method.store_mut
    pub unsafe fn try_execute_mut<A>(&mut self, arg: &mut A) -> bool {
        if self.is_mut() {
            self.execute_mut(arg);
            true
        } else {
            false
        }
    }

    /// If the [`holder`] is not empty and was stored via [`once_mut`] \ [`store_once_mut`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// The [`holder`] becomes empty / [`cleared`] if the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once_mut`] \ [`store_once_mut`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once_mut`]: #method.once_mut
    /// [`store_once_mut`]: #method.store_once_mut
    /// [`cleared`]: #method.clear
    pub unsafe fn try_execute_once_mut<A>(&mut self, arg: &mut A) -> bool {
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
    /// Panics if the [`holder`] is empty.
    /// Panics if the closure was stored via [`new`] \ [`store`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn execute<A>(&mut self, arg: &A) {
        match Self::executor_tag(self.vt) {
            ExecutorTag::None => panic!("tried to execute an empty closure"),
            ExecutorTag::FnOnce => {
                panic!("tried to execute an `FnOnce` immutable arg closure via `execute`")
            }
            ExecutorTag::FnMut => {
                panic!("tried to execute an `FnMut` mutable arg closure via `execute`")
            }
            ExecutorTag::FnOnceMut => {
                panic!("tried to execute an `FnOnce` mutable arg closure via `execute`")
            }
            ExecutorTag::Fn => {}
        };

        self.execute_unchecked(arg);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`new`] \ [`store`].
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`store`];
    /// that the [`holder`] is not empty;
    /// that the closure was not stored via [`new`] \ [`store`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn execute_unchecked<A>(&mut self, arg: &A) {
        (self.vt.execute)(self, arg as *const _ as *mut _);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once`] \ [`store_once`].
    ///
    /// The [`holder`] becomes empty / [`cleared`] after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once`] \ [`store_once`].
    ///
    /// # Panics
    ///
    /// Panics if the [`holder`] is empty.
    /// Panics if the closure was not stored via [`once`] \ [`store_once`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    /// [`cleared`]: #method.clear
    pub unsafe fn execute_once<A>(&mut self, arg: &A) {
        match Self::executor_tag(self.vt) {
            ExecutorTag::None => panic!("tried to execute an empty closure"),
            ExecutorTag::Fn => {
                panic!("tried to execute an `FnMut` immutable arg closure via `execute_once`")
            }
            ExecutorTag::FnMut => {
                panic!("tried to execute an `FnMut` mutable arg closure via `execute_once`")
            }
            ExecutorTag::FnOnceMut => {
                panic!("tried to execute an `FnOnce` mutable arg closure via `execute_once`")
            }
            ExecutorTag::FnOnce => {}
        };

        self.execute_once_unchecked(arg);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once`] \ [`store_once`].
    ///
    /// The [`holder`] becomes empty / [`cleared`] after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once`] \ [`store_once`];
    /// that the [`holder`] is not empty;
    /// that the closure was stored via [`once`] \ [`store_once`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    /// [`cleared`]: #method.clear
    pub unsafe fn execute_once_unchecked<A>(&mut self, arg: &A) {
        (self.vt.execute)(self, arg as *const _ as *mut _);
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
    /// Panics if the [`holder`] is empty.
    /// Panics if the closure was not stored via [`new_mut`] \ [`store_mut`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new_mut`]: #method.new_mut
    /// [`store_mut`]: #method.store_mut
    pub unsafe fn execute_mut<A>(&mut self, arg: &mut A) {
        match Self::executor_tag(self.vt) {
            ExecutorTag::None => panic!("tried to execute an empty closure"),
            ExecutorTag::Fn => {
                panic!("tried to execute an `FnMut` immutable arg closure via `execute_mut`")
            }
            ExecutorTag::FnOnce => {
                panic!("tried to execute an `FnOnce` immutable arg closure via `execute_mut`")
            }
            ExecutorTag::FnOnceMut => {
                panic!("tried to execute an `FnOnce` mutable arg closure via `execute_mut`")
            }
            ExecutorTag::FnMut => {}
        };

        self.execute_mut_unchecked(arg);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`new_mut`] \ [`store_mut`].
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new_mut`] \ [`store_mut`];
    /// that the [`holder`] is not empty;
    /// that the closure was stored via [`new_mut`] \ [`store_mut`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`new_mut`]: #method.new_mut
    /// [`store_mut`]: #method.store_mut
    pub unsafe fn execute_mut_unchecked<A>(&mut self, arg: &mut A) {
        (self.vt.execute)(self, arg as *mut _ as *mut _);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// The [`holder`] becomes empty / [`cleared`] after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once_mut`] \ [`store_once_mut`].
    ///
    /// # Panics
    ///
    /// Panics if the [`holder`] is empty.
    /// Panics if the closure was not stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once_mut`]: #method.once_mut
    /// [`store_once_mut`]: #method.store_once_mut
    /// [`cleared`]: #method.clear
    pub unsafe fn execute_once_mut<A>(&mut self, arg: &mut A) {
        match Self::executor_tag(self.vt) {
            ExecutorTag::None => panic!("tried to execute an empty closure"),
            ExecutorTag::Fn => {
                panic!("tried to execute an `FnMut` immutable arg closure via `execute_once_mut`")
            }
            ExecutorTag::FnMut => {
                panic!("tried to execute an `FnMut` mutable arg closure via `execute_once_mut`")
            }
            ExecutorTag::FnOnce => {
                panic!("tried to execute an `FnOnce` immutable arg closure via `execute_once_mut`")
            }
            ExecutorTag::FnOnceMut => {}
        };

        self.execute_once_mut_unchecked(arg);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// The [`holder`] becomes empty / [`cleared`] after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once_mut`] \ [`store_once_mut`];
    /// that the [`holder`] is not empty;
    /// that the closure was stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// [`holder`]: struct.ClosureHolder.html
    /// [`once_mut`]: #method.once_mut
    /// [`store_once_mut`]: #method.store_once_mut
    /// [`cleared`]: #method.clear
    pub unsafe fn execute_once_mut_unchecked<A>(&mut self, arg: &mut A) {
        (self.vt.execute)(self, arg as *mut _ as *mut _);
    }

    /// Clears the [`holder`].
    ///
    /// Drops the closure captures; frees the allocated storage buffer, if necessary.
    ///
    /// [`holder`]: struct.ClosureHolder.html
    pub unsafe fn clear(&mut self) {
        (self.vt.drop)(self);
    }

    unsafe fn store_closure<F: Sized>(storage: &mut StorageUnion, f: F) {
        let size = mem::size_of::<F>();

        if size > STORAGE_SIZE {
            let ptr = Box::into_raw(vec![0u8; size].into_boxed_slice());

            ptr::write_unaligned::<F>((*ptr).as_mut_ptr() as *mut _, f);

            storage._dynamic = DynamicStorage(ptr);
        } else {
            ptr::write_unaligned::<F>(storage._static.0.as_mut_ptr() as *mut _, f);
        }
    }

    unsafe fn store_impl<F: Sized>(&mut self, vt: &'static ClosureVTable, f: F) {
        assert!(
            self.is_none(),
            "tried to store a closure in an occupied `ClosureHolder`"
        );

        self.vt = vt;
        Self::store_closure(&mut self.storage, f);
    }

    /// Frees allocated dynamic storage memory, if any; clears the vtable and the storage.
    unsafe fn clear_impl(&mut self, storage_tag: StorageTag) {
        if storage_tag == StorageTag::Dynamic {
            Box::from_raw(Self::storage(storage_tag, &mut self.storage) as *mut _);
            // Boxed storage dropped here.
        }

        self.vt = default_vtable();
        self.storage = Default::default();
    }

    unsafe fn closure<F>(storage: &mut [u8]) -> &mut F {
        &mut *(storage.as_mut_ptr() as *mut _)
    }

    unsafe fn take_closure<F>(storage: &mut [u8]) -> F {
        ptr::read_unaligned::<F>(storage.as_mut_ptr() as *mut _)
    }

    unsafe fn arg<'a, A>(arg: *mut ()) -> &'a A {
        &mut *(arg as *mut _)
    }

    unsafe fn arg_mut<'a, A>(arg: *mut ()) -> &'a mut A {
        &mut *(arg as *mut _)
    }

    unsafe fn storage<'s>(tag: StorageTag, storage: &'s mut StorageUnion) -> &'s mut [u8] {
        match tag {
            StorageTag::Static => storage._static.0.as_mut(),
            StorageTag::Dynamic => &mut *storage._dynamic.0,
        }
    }

    #[cfg(test)]
    fn storage_tag(vt: &'static ClosureVTable) -> StorageTag {
        (vt.storage_tag)()
    }

    fn executor_tag(vt: &'static ClosureVTable) -> ExecutorTag {
        (vt.executor_tag)()
    }

    #[cfg(test)]
    fn is_static(&self) -> bool {
        assert!(!self.is_none());
        matches!(Self::storage_tag(self.vt), StorageTag::Static)
    }

    #[cfg(test)]
    fn is_dynamic(&self) -> bool {
        assert!(!self.is_none());
        matches!(Self::storage_tag(self.vt), StorageTag::Dynamic)
    }
}

impl Default for ClosureHolder {
    fn default() -> Self {
        Self::empty()
    }
}

impl Drop for ClosureHolder {
    fn drop(&mut self) {
        unsafe {
            self.clear();
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
    #[should_panic(expected = "tried to store a closure in an occupied `ClosureHolder`")]
    fn double_store() {
        let mut h = unsafe { ClosureHolder::new(|_arg: &usize| println!("Hello")) };

        unsafe {
            h.store(|_arg: &usize| println!(" world!"));
        }
    }

    #[test]
    #[should_panic(expected = "tried to store a closure in an occupied `ClosureHolder`")]
    fn double_store_mut() {
        let mut h = unsafe { ClosureHolder::new_mut(|_arg: &mut usize| println!("Hello")) };

        unsafe {
            h.store_mut(|_arg: &mut usize| println!(" world!"));
        }
    }

    #[test]
    #[should_panic(expected = "tried to store a closure in an occupied `ClosureHolder`")]
    fn double_store_once() {
        let mut h = unsafe { ClosureHolder::once(|_arg: &usize| println!("Hello")) };

        unsafe {
            h.store_once(|_arg: &usize| println!(" world!"));
        }
    }

    #[test]
    #[should_panic(expected = "tried to store a closure in an occupied `ClosureHolder`")]
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
        let capture = [0u8; STORAGE_SIZE];

        let h = unsafe { ClosureHolder::once(move |_arg: &usize| println!("{}", capture.len())) };

        assert!(h.is_static());
    }

    #[test]
    fn dynamic_storage() {
        let capture = [0u8; STORAGE_SIZE + 1];

        let h = unsafe { ClosureHolder::once(move |_arg: &usize| println!("{}", capture.len())) };

        assert!(h.is_dynamic());
    }

    #[test]
    #[should_panic(expected = "tried to execute an empty closure")]
    fn execute_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute(&0usize);
        }
    }

    #[test]
    #[should_panic(expected = "tried to execute an `FnOnce` immutable arg closure via `execute`")]
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
    #[should_panic(expected = "tried to execute an `FnMut` mutable arg closure via `execute`")]
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
    #[should_panic(expected = "tried to execute an `FnOnce` mutable arg closure via `execute`")]
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
    #[should_panic(expected = "tried to execute an empty closure")]
    fn execute_once_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute_once(&0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "tried to execute an `FnMut` immutable arg closure via `execute_once`"
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
    #[should_panic(expected = "tried to execute an `FnMut` mutable arg closure via `execute_once`")]
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
        expected = "tried to execute an `FnOnce` mutable arg closure via `execute_once`"
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
    #[should_panic(expected = "tried to execute an empty closure")]
    fn execute_mut_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "tried to execute an `FnMut` immutable arg closure via `execute_mut`"
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
        expected = "tried to execute an `FnOnce` immutable arg closure via `execute_mut`"
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
    #[should_panic(expected = "tried to execute an `FnOnce` mutable arg closure via `execute_mut`")]
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
    #[should_panic(expected = "tried to execute an empty closure")]
    fn execute_once_mut_empty() {
        let mut h = ClosureHolder::empty();

        unsafe {
            h.execute_once_mut(&mut 0usize);
        }
    }

    #[test]
    #[should_panic(
        expected = "tried to execute an `FnMut` immutable arg closure via `execute_once_mut`"
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
        expected = "tried to execute an `FnMut` mutable arg closure via `execute_once_mut`"
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
        expected = "tried to execute an `FnOnce` immutable arg closure via `execute_once_mut`"
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

    #[test]
    fn destructor_static() {
        static mut NUM_RESOURCES: u32 = 0;

        struct Resource {}

        impl Resource {
            fn new() -> Self {
                unsafe {
                    NUM_RESOURCES += 1;
                }

                Self {}
            }

            fn foo(&self) {
                println!("Hello from resource.");
            }

            fn num_resources() -> u32 {
                unsafe { NUM_RESOURCES }
            }
        }

        impl Drop for Resource {
            fn drop(&mut self) {
                unsafe {
                    debug_assert!(NUM_RESOURCES > 0);
                    NUM_RESOURCES -= 1;
                }
            }
        }

        assert_eq!(Resource::num_resources(), 0);

        // `Fn` closures - cleaned up when dropped.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::new(move |_arg: &usize| {
                    res.foo();
                });
                assert!(h.is_static());

                assert_eq!(Resource::num_resources(), 1);

                h.execute(&0usize);
                h.execute(&0usize);

                assert_eq!(Resource::num_resources(), 1);

                // Closure dropped here, drop handler called.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnMut` closures - cleaned up when dropped.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::new_mut(move |_arg: &mut usize| {
                    res.foo();
                });
                assert!(h.is_static());

                assert_eq!(Resource::num_resources(), 1);

                h.execute_mut(&mut 0usize);
                h.execute_mut(&mut 0usize);

                assert_eq!(Resource::num_resources(), 1);

                // Closure dropped here, drop handler called.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnce` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();
                });
                assert!(h.is_static());

                assert_eq!(Resource::num_resources(), 1);

                h.execute_once(&0usize); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnceMut` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();
                });
                assert!(h.is_static());

                assert_eq!(Resource::num_resources(), 1);

                h.execute_once_mut(&mut 0usize); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnce` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();
                });
                assert!(h.is_static());

                assert_eq!(Resource::num_resources(), 1);

                std::mem::drop(h); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnceMut` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();
                });
                assert!(h.is_static());

                assert_eq!(Resource::num_resources(), 1);

                std::mem::drop(h); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        assert_eq!(Resource::num_resources(), 0);
    }

    #[test]
    fn destructor_dynamic() {
        static mut NUM_RESOURCES: u32 = 0;

        struct Resource {}

        impl Resource {
            fn new() -> Self {
                unsafe {
                    NUM_RESOURCES += 1;
                }

                Self {}
            }

            fn foo(&self) {
                println!("Hello from resource.");
            }

            fn num_resources() -> u32 {
                unsafe { NUM_RESOURCES }
            }
        }

        impl Drop for Resource {
            fn drop(&mut self) {
                unsafe {
                    debug_assert!(NUM_RESOURCES > 0);
                    NUM_RESOURCES -= 1;
                }
            }
        }

        assert_eq!(Resource::num_resources(), 0);

        let capture = [0u8; HOLDER_SIZE];

        // `Fn` closures - cleaned up when dropped.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::new(move |_arg: &usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                assert_eq!(Resource::num_resources(), 1);

                h.execute(&0usize);
                h.execute(&0usize);

                assert_eq!(Resource::num_resources(), 1);

                // Closure dropped here, drop handler called, boxed storage freed.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnMut` closures - cleaned up when dropped.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::new_mut(move |_arg: &mut usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                assert_eq!(Resource::num_resources(), 1);

                h.execute_mut(&mut 0usize);
                h.execute_mut(&mut 0usize);

                assert_eq!(Resource::num_resources(), 1);

                // Closure dropped here, drop handler called, boxed storage freed.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnce` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                assert_eq!(Resource::num_resources(), 1);

                h.execute_once(&0usize); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnceMut` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                assert_eq!(Resource::num_resources(), 1);

                h.execute_once_mut(&mut 0usize); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnce` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                assert_eq!(Resource::num_resources(), 1);

                std::mem::drop(h); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        // `FnOnceMut` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                assert_eq!(Resource::num_resources(), 1);

                std::mem::drop(h); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);
            }

            assert_eq!(Resource::num_resources(), 0);
        }

        assert_eq!(Resource::num_resources(), 0);
    }

    #[test]
    fn destructor_once_static() {
        static mut NUM_RESOURCES: u32 = 0;

        struct Resource {}

        impl Resource {
            fn new() -> Self {
                unsafe {
                    NUM_RESOURCES += 1;
                }

                Self {}
            }

            fn foo(&self) {
                println!("Hello from resource.");
            }

            fn num_resources() -> u32 {
                unsafe { NUM_RESOURCES }
            }
        }

        impl Drop for Resource {
            fn drop(&mut self) {
                unsafe {
                    debug_assert!(NUM_RESOURCES > 0);
                    NUM_RESOURCES -= 1;
                }
            }
        }

        assert_eq!(Resource::num_resources(), 0);

        // `FnOnce` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();
                });
                assert!(h.is_static());

                // Closure dropped here.
                h.execute_once(&0usize);

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }
        }

        // `FnOnce` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();
                });
                assert!(h.is_static());

                std::mem::drop(h); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);
            }
        }

        // `FnOnceMut` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();
                });
                assert!(h.is_static());

                h.execute_once_mut(&mut 0usize); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }
        }

        // `FnOnceMut` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();
                });
                assert!(h.is_static());

                std::mem::drop(h); // Closure dropped here.

                assert_eq!(Resource::num_resources(), 0);
            }
        }

        assert_eq!(Resource::num_resources(), 0);
    }

    #[test]
    fn destructor_once_dynamic() {
        static mut NUM_RESOURCES: u32 = 0;

        struct Resource {}

        impl Resource {
            fn new() -> Self {
                unsafe {
                    NUM_RESOURCES += 1;
                }

                Self {}
            }

            fn foo(&self) {
                println!("Hello from resource.");
            }

            fn num_resources() -> u32 {
                unsafe { NUM_RESOURCES }
            }
        }

        impl Drop for Resource {
            fn drop(&mut self) {
                unsafe {
                    debug_assert!(NUM_RESOURCES > 0);
                    NUM_RESOURCES -= 1;
                }
            }
        }

        assert_eq!(Resource::num_resources(), 0);

        // `FnOnce` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            let capture = [0u8; HOLDER_SIZE];

            unsafe {
                let mut h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                h.execute_once(&0usize); // Closure dropped here, boxed storage freed.

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }
        }

        // `FnOnce` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            let capture = [0u8; HOLDER_SIZE];

            unsafe {
                let h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                std::mem::drop(h); // Closure dropped here, boxed storage freed.

                assert_eq!(Resource::num_resources(), 0);
            }
        }

        // `FnOnceMut` closures - cleaned up when executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            let capture = [0u8; HOLDER_SIZE];

            unsafe {
                let mut h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                h.execute_once_mut(&mut 0usize); // Closure dropped here, boxed storage freed.

                assert_eq!(Resource::num_resources(), 0);

                std::mem::drop(h); // Safe to drop - there won't be a double free, as we fixed up the vtable after closure execution.
            }
        }

        // `FnOnceMut` closures - cleaned up when dropped without being executed.
        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            let capture = [0u8; HOLDER_SIZE];

            unsafe {
                let h = ClosureHolder::once_mut(move |_arg: &mut usize| {
                    res.foo();

                    println!("{}", capture.len());
                });
                assert!(h.is_dynamic());

                std::mem::drop(h); // Closure dropped here, boxed storage freed.

                assert_eq!(Resource::num_resources(), 0);
            }
        }

        assert_eq!(Resource::num_resources(), 0);
    }
}
