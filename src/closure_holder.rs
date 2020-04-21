use std::mem::{self, size_of};
use std::ptr;

use num_derive::{FromPrimitive, ToPrimitive};
use static_assertions::{assert_eq_size, const_assert_eq};
use strum_macros::EnumCount;

/// Static function which
/// 1) unpacks the correct closure type from the raw byte buffer,
/// 2) casts the argument ptr to the expected argument type,
/// 3) executes the closure.
///
/// Gets passed the raw closure capture buffer from the [`ClosureHolder`], static or dynamic,
/// and the pointer to type-erased closure argument.
/// It's up to the user to ensure the argument is of type the closure expects.
///
/// [`ClosureHolder`]: #struct.ClosureHolder.html
type Executor = fn(&mut [u8], *mut ());

/// Static function which
/// 1) unpacks the correct closure type from the raw byte buffer,
/// 2) drops it.
///
/// Gets passed the raw closure capture buffer from the [`ClosureHolder`], static or dynamic.
///
/// [`ClosureHolder`]: #struct.ClosureHolder.html
type DropHandler = fn(&mut [u8]);

/// Total amount of memory used by the [`ClosureHolder`]
///
/// [`ClosureHolder`]: #struct.ClosureHolder.html
const HOLDER_SIZE: usize = 64;

const EXECUTOR_SIZE: usize = size_of::<Executor>();
const DROP_HANDLER_SIZE: usize = size_of::<DropHandler>();
const DISCRIMINANT_SIZE: usize = size_of::<u8>();

/// Amount of memory left for closure capture storage left in the [`ClosureHolder`].
///
/// x86: 4b (executor func ptr) + 4b (drop handler func ptr) + 1b (discriminant) + 55b (closure storage) = 64b
/// x64: 8b (executor func ptr) + 8b (drop handler func ptr) + 1b (discriminant) + 47b (closure storage) = 64b
///
/// [`ClosureHolder`]: #struct.ClosureHolder.html
const STORAGE_SIZE: usize =
    HOLDER_SIZE /* 64b */
    - EXECUTOR_SIZE /* 4b / 8b -> 60b / 56b */
    - DISCRIMINANT_SIZE /* 1b -> 59b / 55b */
    - DROP_HANDLER_SIZE /* 4b / 8b -> 55b / 47b */;

/// Raw byte buffer used to store the closure capture, if it fits in `STORAGE_SIZE` bytes.
#[repr(C, packed(1))]
#[derive(Clone, Copy)] // `Copy` needed for union storage.
struct StaticStorage([u8; STORAGE_SIZE]);

const_assert_eq!(size_of::<StaticStorage>(), STORAGE_SIZE);

/// Heap-allocated buffer used to store the closure capture, if it does not fit in `STORAGE_SIZE` bytes.
#[repr(C, packed(1))]
#[derive(Clone, Copy)] // `Copy` needed for union storage.
struct DynamicStorage(*mut [u8]);

const_assert_eq!(size_of::<DynamicStorage>(), size_of::<usize>() * 2); // Fat pointer.

/// Closure capture storage.
/// Static or dynamic.
/// Tag/discriminant is encoded separately.
#[repr(C, packed(1))]
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

/// Explicit storage union tag, encoded separately.
#[derive(Clone, Copy, Eq, Debug, PartialEq, EnumCount, FromPrimitive, ToPrimitive)]
enum StorageTag {
    Static = 0,
    Dynamic,
}

/// Explicit executor union tag, encoded separately.
#[derive(EnumCount, FromPrimitive, ToPrimitive)]
enum ExecutorTag {
    None = 0,
    /// `FnMut` closure. Single immutable ref arg. Executes multiple times.
    //Fn(ClosureExecutorFn),
    Fn,
    /// `FnOnce` closure. Single immutable ref arg. Executes once.
    //FnOnce(ClosureExecutorFn),
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

const STORAGE_TAG_OFFSET: u8 = 0;
const STORAGE_TAG_BITS: u8 = num_bits(STORAGETAG_COUNT as u8);
const_assert_eq!(STORAGE_TAG_BITS, 1);
const STORAGE_TAG_MASK: u8 = (1 << STORAGE_TAG_BITS) - 1;

const EXECUTOR_TAG_OFFSET: u8 = STORAGE_TAG_OFFSET + STORAGE_TAG_BITS;
const EXECUTOR_TAG_BITS: u8 = num_bits(EXECUTORTAG_COUNT as u8);
const_assert_eq!(EXECUTOR_TAG_BITS, 3);
const EXECUTOR_TAG_MASK: u8 = (1 << EXECUTOR_TAG_BITS) - 1;

/// Closure holder discriminant.
/// Encodes storage union tag and executor union tag.
///
/// . . . . e e e s
#[repr(transparent)]
#[derive(Clone, Copy)]
struct Discriminant(u8);

impl Discriminant {
    fn new(storage: StorageTag, executor: ExecutorTag) -> Self {
        use num_traits::ToPrimitive;
        Self(
            (storage.to_u8().unwrap() & STORAGE_TAG_MASK) << STORAGE_TAG_OFFSET
                | (executor.to_u8().unwrap() & EXECUTOR_TAG_MASK) << EXECUTOR_TAG_OFFSET,
        )
    }

    fn storage_tag(self) -> StorageTag {
        use num_traits::FromPrimitive;
        StorageTag::from_u8(read_bits(self.0, STORAGE_TAG_BITS, STORAGE_TAG_OFFSET)).unwrap()
    }

    fn executor_tag(self) -> ExecutorTag {
        use num_traits::FromPrimitive;
        ExecutorTag::from_u8(read_bits(self.0, EXECUTOR_TAG_BITS, EXECUTOR_TAG_OFFSET)).unwrap()
    }

    fn set_executor_tag(&mut self, executor: ExecutorTag) {
        *self = Discriminant::new(self.storage_tag(), executor);
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
/// borrows live until the call to [`execute`] \ [`execute_mut`] \ [`execute_once`] \ [`execute_once_mut`].
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
#[repr(C, packed(1))]
pub struct ClosureHolder {
    executor: Executor,         // offs 0b          size 4b / 8b
    drop_handler: DropHandler,  // offs 4b / 8b     size 4b / 8b
    storage: StorageUnion,      // offs 8b / 16b    size 55b / 47b
    discriminant: Discriminant, // offs 63b         size 1b
}

const_assert_eq!(size_of::<ClosureHolder>(), HOLDER_SIZE);

impl ClosureHolder {
    /// Creates an empty [`ClosureHolder`].
    ///
    /// [`ClosureHolder`]: #struct.ClosureHolder.html
    pub fn empty() -> Self {
        ClosureHolder {
            executor: |_, _| {},
            drop_handler: |_| {},
            storage: Default::default(),
            discriminant: Discriminant::new(StorageTag::Static, ExecutorTag::None),
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
        let mut result = Self::empty();
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
        let mut result = Self::empty();
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
        let mut result = Self::empty();
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
        let mut result = Self::empty();
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
            |storage, arg| {
                (Self::closure::<F>(storage))(Self::arg(arg));
            },
            ExecutorTag::Fn,
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
            |storage, arg| {
                (Self::closure::<F>(storage))(Self::arg(arg));
            },
            ExecutorTag::FnMut,
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
            |storage, arg| {
                (Self::take_closure::<F>(storage))(Self::arg(arg)); // Closure dropped here.
            },
            ExecutorTag::FnOnce,
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
            |storage, arg| {
                (Self::take_closure::<F>(storage))(Self::arg(arg)); // Closure dropped here.
            },
            ExecutorTag::FnOnceMut,
            f,
        );
    }

    /// If the [`ClosureHolder`] is not empty, returns `true`; otherwise returns `false`.
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    pub fn is_none(&self) -> bool {
        self.discriminant.executor_tag().is_none()
    }

    /// If the [`ClosureHolder`] is not empty and was stored via [`new`] \ [`store`],
    /// returns `true` (the user may call [`execute`]); otherwise returns `false`.
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    /// [`execute`]: #method.execute
    pub fn is_some(&self) -> bool {
        self.discriminant.executor_tag().is_some()
    }

    /// If the [`ClosureHolder`] is not empty and was stored via [`once`] \ [`store_once`],
    /// returns `true` (the user may call [`execute_once`]); otherwise returns `false`.
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    /// [`execute_once`]: #method.execute_once
    pub fn is_once(&self) -> bool {
        self.discriminant.executor_tag().is_once()
    }

    /// If the [`ClosureHolder`] is not empty and was stored via [`new_mut`] \ [`store_mut`],
    /// returns `true` (the user may call [`execute_mut`]); otherwise returns `false`.
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`new_mut`]: #method.new
    /// [`store_mut`]: #method.store
    /// [`execute_mut`]: #method.execute_mut
    pub fn is_mut(&self) -> bool {
        self.discriminant.executor_tag().is_mut()
    }

    /// If the [`ClosureHolder`] is not empty and was stored via [`once_mut`] \ [`store_once_mut`],
    /// returns `true` (the user may call [`execute_once_mut`]); otherwise returns `false`.
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`once_mut`]: #method.new
    /// [`store_once_mut`]: #method.store
    /// [`execute_once_mut`]: #method.execute_once_mut
    pub fn is_once_mut(&self) -> bool {
        self.discriminant.executor_tag().is_once_mut()
    }

    /// If the [`ClosureHolder`] is not empty and was stored via [`new`] \ [`store`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`store`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn try_execute<'a, A>(&mut self, arg: &'a A) -> bool {
        if self.is_some() {
            self.execute(arg);
            true
        } else {
            false
        }
    }

    /// If the [`ClosureHolder`] is not empty and was stored via [`once`] \ [`store_once`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// [`ClosureHolder`] becomes empty if the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new`] \ [`once`] \ [`store`] \ [`store_once`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
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

    /// If the [`ClosureHolder`] is not empty and was stored via [`new_mut`] \ [`store_mut`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`new_mut`] \ [`store_mut`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`new_mut`]: #method.new_mut
    /// [`store_mut`]: #method.store_mut
    pub unsafe fn try_execute_mut<'a, A>(&mut self, arg: &'a mut A) -> bool {
        if self.is_mut() {
            self.execute_mut(arg);
            true
        } else {
            false
        }
    }

    /// If the [`ClosureHolder`] is not empty and was stored via [`once_mut`] \ [`store_once_mut`],
    /// executes the stored closure and returns `true`; otherwise returns `false`.
    ///
    /// [`ClosureHolder`] becomes empty if the closure is executed.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once_mut`] \ [`store_once_mut`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
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
    /// Panics if the [`ClosureHolder`] is empty.
    /// Panics if the closure was not stored via [`new`] \ [`store`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`new`]: #method.new
    /// [`store`]: #method.store
    pub unsafe fn execute<'a, A>(&mut self, arg: &'a A) {
        match self.discriminant.executor_tag() {
            ExecutorTag::None => panic!("Tried to execute an empty closure."),
            ExecutorTag::FnOnce => {
                panic!("Tried to execute an `FnOnce` immutable arg closure via `execute`.")
            }
            ExecutorTag::FnMut => {
                panic!("Tried to execute an `FnMut` mutable arg closure via `execute`.")
            }
            ExecutorTag::FnOnceMut => {
                panic!("Tried to execute an `FnOnce` mutable arg closure via `execute`.")
            }
            ExecutorTag::Fn => {}
        };

        (self.executor)(self.storage(), arg as *const _ as *mut _);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once`] \ [`store_once`].
    ///
    /// The [`ClosureHolder`] becomes empty / [`cleared`] after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once`] \ [`store_once`].
    ///
    /// # Panics
    ///
    /// Panics if the [`ClosureHolder`] is empty.
    /// Panics if the closure was not stored via [`once`] \ [`store_once`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`once`]: #method.once
    /// [`store_once`]: #method.store_once
    /// [`cleared`]: #method.clear
    pub unsafe fn execute_once<'a, A>(&mut self, arg: &'a A) {
        match self.discriminant.executor_tag() {
            ExecutorTag::None => panic!("Tried to execute an empty closure."),
            ExecutorTag::Fn => {
                panic!("Tried to execute an `FnMut` immutable arg closure via `execute_once`.")
            }
            ExecutorTag::FnMut => {
                panic!("Tried to execute an `FnMut` mutable arg closure via `execute_once`.")
            }
            ExecutorTag::FnOnceMut => {
                panic!("Tried to execute an `FnOnce` mutable arg closure via `execute_once`.")
            }
            ExecutorTag::FnOnce => {}
        };

        (self.executor)(self.storage(), arg as *const _ as *mut _);

        // Do not run the drop handler in `clear`.
        self.discriminant.set_executor_tag(ExecutorTag::None);

        self.clear();
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
    /// Panics if the [`ClosureHolder`] is empty.
    /// Panics if the closure was not stored via [`new_mut`] \ [`store_mut`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`new_mut`]: #method.new_mut
    /// [`store_mut`]: #method.store_mut
    pub unsafe fn execute_mut<'a, A>(&mut self, arg: &'a mut A) {
        match self.discriminant.executor_tag() {
            ExecutorTag::None => panic!("Tried to execute an empty closure."),
            ExecutorTag::Fn => {
                panic!("Tried to execute an `FnMut` immutable arg closure via `execute_mut`.")
            }
            ExecutorTag::FnOnce => {
                panic!("Tried to execute an `FnOnce` immutable arg closure via `execute_mut`.")
            }
            ExecutorTag::FnOnceMut => {
                panic!("Tried to execute an `FnOnce` mutable arg closure via `execute_mut`.")
            }
            ExecutorTag::FnMut => {}
        };

        (self.executor)(self.storage(), arg as *mut _ as *mut _);
    }

    /// Executes the stored closure unconditionally.
    ///
    /// May only be called for closures stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// The [`ClosureHolder`] becomes empty / [`cleared`] after this call.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the function is passed the same argument type
    /// as the one used in the previous call to [`once_mut`] \ [`store_once_mut`].
    ///
    /// # Panics
    ///
    /// Panics if the [`ClosureHolder`] is empty.
    /// Panics if the closure was not stored via [`once_mut`] \ [`store_once_mut`].
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    /// [`once_mut`]: #method.once_mut
    /// [`store_once_mut`]: #method.store_once_mut
    /// [`cleared`]: #method.clear
    pub unsafe fn execute_once_mut<'a, A>(&mut self, arg: &'a mut A) {
        match self.discriminant.executor_tag() {
            ExecutorTag::None => panic!("Tried to execute an empty closure."),
            ExecutorTag::Fn => {
                panic!("Tried to execute an `FnMut` immutable arg closure via `execute_once_mut`.")
            }
            ExecutorTag::FnMut => {
                panic!("Tried to execute an `FnMut` mutable arg closure via `execute_once_mut`.")
            }
            ExecutorTag::FnOnce => {
                panic!("Tried to execute an `FnOnce` immutable arg closure via `execute_once_mut`.")
            }
            ExecutorTag::FnOnceMut => {}
        };

        (self.executor)(self.storage(), arg as *mut _ as *mut _);

        // Do not run the drop handler in `clear`.
        self.discriminant.set_executor_tag(ExecutorTag::None);

        self.clear();
    }

    /// Clears the [`ClosureHolder`].
    ///
    /// Drops the closure captures; frees the allocated storage buffer, if necessary.
    ///
    /// [`ClosureHolder`]: struct.ClosureHolder.html
    pub unsafe fn clear(&mut self) {
        // `FnMut` closures need a drop handler.
        // `FnOnce` closures are dropped when executed.
        match self.discriminant.executor_tag() {
            ExecutorTag::Fn | ExecutorTag::FnMut => {
                (self.drop_handler)(Self::storage(self));
            }
            ExecutorTag::FnOnce | ExecutorTag::FnOnceMut => {}
            ExecutorTag::None => {}
        }

        if self.discriminant.storage_tag() == StorageTag::Dynamic {
            let storage = Box::from_raw(self.storage._dynamic.0);
            mem::drop(storage);
        }

        self.executor = |_, _| {};
        self.drop_handler = |_| {};

        self.storage = Default::default();

        self.discriminant = Discriminant::new(StorageTag::Static, ExecutorTag::None);
    }

    unsafe fn store_closure<'a, F: Sized>(storage: &mut StorageUnion, f: F) -> StorageTag {
        let size = mem::size_of::<F>();

        if size > STORAGE_SIZE {
            let ptr = Box::into_raw(vec![0u8; size].into_boxed_slice());

            ptr::write::<F>((*ptr).as_mut_ptr() as *mut _, f);

            storage._dynamic = DynamicStorage(ptr);

            StorageTag::Dynamic
        } else {
            ptr::write::<F>(storage._static.0.as_mut_ptr() as *mut _, f);

            StorageTag::Static
        }
    }

    unsafe fn store_impl<F: Sized>(&mut self, executor: Executor, executor_tag: ExecutorTag, f: F) {
        assert!(
            self.is_none(),
            "Tried to store a closure in an occupied `ClosureHolder`."
        );

        self.executor = executor;

        // `FnMut` closures need a drop handler.
        // `FnOnce` closures are dropped when executed.
        match executor_tag {
            ExecutorTag::Fn | ExecutorTag::FnMut => {
                self.drop_handler = |storage| {
                    let f = Self::take_closure::<F>(storage);
                    mem::drop(f);
                };
            }
            ExecutorTag::FnOnce | ExecutorTag::FnOnceMut => {}
            ExecutorTag::None => unreachable!(),
        }

        let storage_tag = Self::store_closure(&mut self.storage, f);

        self.discriminant = Discriminant::new(storage_tag, executor_tag);
    }

    unsafe fn closure<F>(storage: &mut [u8]) -> &mut F {
        &mut *(storage.as_mut_ptr() as *mut _)
    }

    unsafe fn take_closure<F>(storage: &mut [u8]) -> F {
        ptr::read::<F>(storage.as_mut_ptr() as *mut _)
    }

    unsafe fn arg<'a, A>(arg: *mut ()) -> &'a mut A {
        &mut *(arg as *mut _)
    }

    unsafe fn storage(&mut self) -> &mut [u8] {
        match self.discriminant.storage_tag() {
            StorageTag::Static => self.storage._static.0.as_mut(),
            StorageTag::Dynamic => &mut *self.storage._dynamic.0,
        }
    }

    #[cfg(test)]
    fn is_static(&self) -> bool {
        assert!(!self.is_none());

        if self.discriminant.storage_tag() == StorageTag::Static {
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    fn is_dynamic(&self) -> bool {
        assert!(!self.is_none());

        if self.discriminant.storage_tag() == StorageTag::Dynamic {
            true
        } else {
            false
        }
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

/// Number of bits needed to represent `count` enum states.
const fn num_bits(count: u8) -> u8 {
    8 - (count - 1).leading_zeros() as u8
}

fn read_bits(source: u8, num_bits_to_read: u8, offset: u8) -> u8 {
    debug_assert!((num_bits_to_read + offset) <= 8);

    let read_mask = (1 << num_bits_to_read) - 1;

    (source >> offset) & read_mask
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

        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::new(move |_arg: &usize| {
                    res.foo();
                });

                h.execute(&0usize);
                h.execute(&0usize);

                assert_eq!(Resource::num_resources(), 1);

                // Closure dropped here, drop handler called.
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

        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            let capture = [0u8; HOLDER_SIZE];

            unsafe {
                let mut h = ClosureHolder::new(move |_arg: &usize| {
                    res.foo();

                    println!("{}", capture.len());
                });

                h.execute(&0usize);
                h.execute(&0usize);

                assert_eq!(Resource::num_resources(), 1);

                // Closure dropped here, drop handler called, boxed storage freed.
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

        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            unsafe {
                let mut h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();
                });

                // Closure dropped here.
                h.execute_once(&0usize);

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

        {
            let res = Resource::new();
            assert_eq!(Resource::num_resources(), 1);

            let capture = [0u8; HOLDER_SIZE];

            unsafe {
                let mut h = ClosureHolder::once(move |_arg: &usize| {
                    res.foo();

                    println!("{}", capture.len());
                });

                // Closure dropped here, boxed storage freed.
                h.execute_once(&0usize);

                assert_eq!(Resource::num_resources(), 0);
            }
        }

        assert_eq!(Resource::num_resources(), 0);
    }
}
