ulong Scalar64_mac_with_carry(ulong a, ulong b, ulong c, ulong *d) {
    printf("vmx: mac_with_carry: a, b, c, d: %016lx %016lx %016lx %016lx\n", a, b, c, *d);
    ulong lo = a * b + c;
    printf("vmx: mac_with_carry: lo: %016lx\n", lo);
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    printf("vmx: mac_with_carry: hi: %016lx\n", hi);
    *d = hi;
    return lo;
}

__kernel void test_mul_64(__global ulong *result) {
    ulong carry = 0;
    ulong low = Scalar64_mac_with_carry(0x00000003fffffffc, 0xffffffff00000001, 0xfffffff800000004, &carry);
    printf("vmx: carry: %016lx\n", carry);
    printf("vmx: low %016lx\n", low);
    *result = carry;
}
