//#[cfg(feature = "gpu")]
// TODO vmx 2021-04-30: FFT is still using opencl even when CUDA is used for multiexp
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
#[cfg(any(feature = "opencl", feature = "cuda"))]
use rust_gpu_tools::opencl;

#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("GPUError: {0}")]
    Simple(&'static str),
    //#[cfg(feature = "gpu")]
    // TODO vmx 2021-04-30: FFT is still using opencl even when CUDA is used for multiexp
    #[cfg(any(feature = "opencl", feature = "cuda"))]
    #[error("OpenCL Error: {0}")]
    OpenCL(#[from] opencl::GPUError),
    #[cfg(feature = "gpu")]
    #[error("GPU taken by a high priority process!")]
    GPUTaken,
    #[cfg(feature = "gpu")]
    #[error("No kernel is initialized!")]
    KernelUninitialized,
    #[error("GPU accelerator is disabled!")]
    GPUDisabled,
    #[cfg(feature = "cuda")]
    #[error("Cuda Error: {0}")]
    Cuda(#[from] cuda::GPUError),
}

pub type GPUResult<T> = std::result::Result<T, GPUError>;

#[cfg(feature = "gpu")]
impl From<std::boxed::Box<dyn std::any::Any + std::marker::Send>> for GPUError {
    fn from(e: std::boxed::Box<dyn std::any::Any + std::marker::Send>) -> Self {
        match e.downcast::<Self>() {
            Ok(err) => *err,
            Err(_) => GPUError::Simple("An unknown GPU error happened!"),
        }
    }
}
