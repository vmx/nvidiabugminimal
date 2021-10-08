use rust_gpu_tools::{opencl, Device};

static SOURCE: &str = include_str!("kernel.cl");

pub fn main() {
    let aa: u32 = 972342711;
    let bb: u32 = 1698717651;

    let device = *Device::all().first().expect("Cannot get a default device");
    let opencl_device = device.opencl_device().unwrap();
    let program = opencl::Program::from_opencl(opencl_device, SOURCE).unwrap();

    let kernel = program.create_kernel("minimal_two_vars", 1, 1).unwrap();
    kernel.arg(&aa).arg(&bb).run().unwrap();
}
