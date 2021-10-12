__kernel void call_mul_hi(__global ulong *result) {
    *result = mul_hi((ulong)0xff, (ulong)0xff00000000000001);
}
