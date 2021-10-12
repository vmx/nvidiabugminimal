__kernel void test_mul_64(__global ulong *result) {
    *result = mad_hi((ulong)0x00000003fffffffc, (ulong)0xffffffff00000001, (ulong)true);
}
