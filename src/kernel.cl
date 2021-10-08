__kernel void minimal_two_vars(uint aa, uint bb, __global uint *result) {
    //printf("aa: %u\n", aa);
    //printf("bb: %u\n", bb);
    uint one = bb << 1;
    uint two = aa >> 31;
     //With printing this, it's correct, without not.
    //printf("one, two: %u %u\n", one, two);
    //printf("one, two: %u %u\n", aa, bb);
    //printf("one | two: %u\n", one | two);

    *result = one | two;
}
