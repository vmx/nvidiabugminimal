__kernel void test_mul_64(__global ulong *result) {
    *result = mad_hi((ulong)0xff, (ulong)0xff00000000000001, (ulong)0x01);
}
