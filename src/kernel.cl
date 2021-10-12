__kernel void test_mul_64(__global ulong *result) {
    ulong hi = mad_hi((ulong)0x00000003fffffffc, (ulong)0xffffffff00000001, (ulong)true);
    printf("vmx: hi: %016lx\n", hi);
    *result = hi;
}
