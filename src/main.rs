use std::ptr;
use std::fs;

use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::{self, ExecuteKernel},
    memory::{Buffer, CL_MEM_READ_ONLY},
    platform,
    program::Program,
    types::CL_BLOCKING,
};

static SOURCE: &str = include_str!("kernel.cl");
//static SPIRV: &[u8] = include_bytes!("../intel_working.spv");
//static SPIRV: &[u8] = include_bytes!("../intel_broken.spv");
//static SPIRV: &[u8] = include_bytes!("../working.spv");
//static SPIRV: &[u8] = include_bytes!("../broken.spv");
//static SPIRV: &[u8] = include_bytes!("../broken_noprint.spv");
//static SPIRV: &[u8] = include_bytes!("../working/spirv.bin");
//static SPIRV: &[u8] = include_bytes!("../broken/spirv.bin");

pub fn main() {
    let aa: u32 = 972342711;
    let bb: u32 = 1698717651;
    let mut result: [u32; 1] = [0];

    let platform = *platform::get_platforms().unwrap().first().unwrap();
    let raw_device = *platform
        .get_devices(CL_DEVICE_TYPE_GPU)
        .unwrap()
        .first()
        .unwrap();
    let device = Device::new(raw_device);
    let context = Context::from_device(&device).unwrap();
    let mut program = Program::create_from_source(&context, SOURCE).unwrap();
    //let mut program = Program::create_from_il(&context, SPIRV).unwrap();
    if let Err(_) = program.build(context.devices(), "-cl-opt-disable") {
        let log = program.get_build_log(context.devices()[0]).unwrap();
        println!("error: {}", log);
    }

    let kernels = kernel::create_program_kernels(&program).unwrap();
    let kernel = kernels.first().unwrap();
    let queue = CommandQueue::create_with_properties(&context, raw_device, 0, 0).unwrap();

    let result_buffer =
        Buffer::<u32>::create(&context, CL_MEM_READ_ONLY, 1, ptr::null_mut()).unwrap();

    ExecuteKernel::new(&kernel)
        .set_arg(&aa)
        .set_arg(&bb)
        .set_arg(&result_buffer)
        .set_local_work_size(1)
        .set_global_work_size(1)
        .enqueue_nd_range(&queue)
        .unwrap();

    queue
        .enqueue_read_buffer(&result_buffer, CL_BLOCKING, 0, &mut result, &[])
        .unwrap();

    // Write the current binary for further inspection to disk.
    fs::write("intel.bin", program.get_binaries().unwrap()[0].clone()).unwrap();

    println!("vmx: result: {:?}", result[0]);
}
