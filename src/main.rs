use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{Device, CL_DEVICE_TYPE_GPU},
    kernel::{self, ExecuteKernel},
    platform,
    program::Program,
};

//static SOURCE: &str = include_str!("kernel.cl");
//static SPIRV: &[u8] = include_bytes!("../intel_working.spv");
static SPIRV: &[u8] = include_bytes!("../intel_broken.spv");

pub fn main() {
    let aa: u32 = 972342711;
    let bb: u32 = 1698717651;

    let platform = *platform::get_platforms().unwrap().first().unwrap();
    let raw_device = *platform
        .get_devices(CL_DEVICE_TYPE_GPU)
        .unwrap()
        .first()
        .unwrap();
    let device = Device::new(raw_device);
    let context = Context::from_device(&device).unwrap();
    //let program = Program::create_and_build_from_source(&context, SOURCE, "").unwrap();
    let mut program = Program::create_from_il(&context, SPIRV).unwrap();
    program.build(&[raw_device], "").unwrap();


    let kernels = kernel::create_program_kernels(&program).unwrap();
    let kernel = kernels.first().unwrap();
    let queue = CommandQueue::create_with_properties(&context, raw_device, 0, 0).unwrap();
    ExecuteKernel::new(&kernel)
        .set_arg(&aa)
        .set_arg(&bb)
        .set_local_work_size(1)
        .set_global_work_size(1)
        .enqueue_nd_range(&queue)
        .unwrap();
}
