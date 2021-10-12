use std::fs;
use std::ptr;

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

pub fn main() {
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
    if program.build(context.devices(), "").is_err() {
    // The bug doesn't happen if the optimizations are turned off.
    //if program.build(context.devices(), "-cl-opt-disable").is_err() {
        let log = program.get_build_log(context.devices()[0]).unwrap();
        println!("error: {}", log);
    }

    let kernels = kernel::create_program_kernels(&program).unwrap();
    let kernel = kernels.first().unwrap();
    let queue = CommandQueue::create_with_properties(&context, raw_device, 0, 0).unwrap();

    let result_buffer =
        Buffer::<u32>::create(&context, CL_MEM_READ_ONLY, 1, ptr::null_mut()).unwrap();

    ExecuteKernel::new(kernel)
        .set_arg(&result_buffer)
        .set_local_work_size(1)
        .set_global_work_size(1)
        .enqueue_nd_range(&queue)
        .unwrap();

    queue
        .enqueue_read_buffer(&result_buffer, CL_BLOCKING, 0, &mut result, &[])
        .unwrap();

    // Write the current binary for further inspection to disk.
    fs::write("kernel.bin", program.get_binaries().unwrap()[0].clone()).unwrap();

    println!("result: {}", result[0]);
    assert_eq!(result[0], 254, "The result is expected to be 254, but it was {}.", result[0]);
}
