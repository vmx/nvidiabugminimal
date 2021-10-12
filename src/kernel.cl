  #define DEVICE
  #define GLOBAL __global
  #define KERNEL __kernel

#define Scalar64_limb ulong
#define Scalar64_LIMBS 4
#define Scalar64_P ((Scalar64){ { 18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352 } })
#define Scalar64_INV 18446744069414584319
typedef struct { Scalar64_limb val[Scalar64_LIMBS]; } Scalar64;

void Scalar64_print(Scalar64 a) {
  printf("0x");
  for (uint i = 0; i < Scalar64_LIMBS; i++) {
    printf("%016lx", a.val[Scalar64_LIMBS - i - 1]);
  }
}

void t_print(Scalar64_limb t[]) {
  printf("0x");
  for (uint i = 0; i < Scalar64_LIMBS + 2; i++) {
    printf("%016lx", t[i]);
  }
}

// Returns a * b + c + d, puts the carry in d
DEVICE ulong Scalar64_mac_with_carry(ulong a, ulong b, ulong c, ulong *d) {
    printf("vmx: mac_with_carry: a, b, c, d: %016lx %016lx %016lx %016lx\n", a, b, c, *d);
    ulong lo = a * b + c;
    printf("vmx: mac_with_carry: lo: %016lx\n", lo);
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    printf("vmx: mac_with_carry: hi: %016lx\n", lo);
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
}

// Modular multiplication
DEVICE Scalar64 Scalar64_mul_default(Scalar64 a, Scalar64 b) {
  Scalar64_limb t[Scalar64_LIMBS + 2] = {0};
  printf("vmx: t1: "); t_print(t); printf("\n");
  uchar i = 0;
    Scalar64_limb carry = 0;
    t[0] = Scalar64_mac_with_carry(a.val[0], b.val[0], t[0], &carry);

    carry = 0;
    Scalar64_limb m = Scalar64_INV * t[0];
    printf("vmx: m: %016lx\n", m);
    Scalar64_mac_with_carry(m, Scalar64_P.val[0], t[0], &carry);
    printf("vmx: carry2: %016lx\n", carry);

  Scalar64 result;
  for(uchar i = 0; i < Scalar64_LIMBS; i++) result.val[i] = t[i];
  return result;
}

KERNEL void test_mul_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
  *result = Scalar64_mul_default(a, b);
}
