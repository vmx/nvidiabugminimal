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

// Returns a + b, puts the carry in d
DEVICE ulong Scalar64_add_with_carry(ulong a, ulong *b) {
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
}

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

// Greater than or equal
DEVICE bool Scalar64_gte(Scalar64 a, Scalar64 b) {
  for(char i = Scalar64_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Normal addition
  DEVICE Scalar64 Scalar64_add_(Scalar64 a, Scalar64 b) {
    bool carry = 0;
    for(uchar i = 0; i < Scalar64_LIMBS; i++) {
      Scalar64_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  Scalar64 Scalar64_sub_(Scalar64 a, Scalar64 b) {
    bool borrow = 0;
    for(uchar i = 0; i < Scalar64_LIMBS; i++) {
      Scalar64_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }

// Modular subtraction
DEVICE Scalar64 Scalar64_sub(Scalar64 a, Scalar64 b) {
  Scalar64 res = Scalar64_sub_(a, b);
  if(!Scalar64_gte(a, b)) res = Scalar64_add_(res, Scalar64_P);
  return res;
}

// Modular addition
DEVICE Scalar64 Scalar64_add(Scalar64 a, Scalar64 b) {
  Scalar64 res = Scalar64_add_(a, b);
  if(Scalar64_gte(res, Scalar64_P)) res = Scalar64_sub_(res, Scalar64_P);
  return res;
}

// Modular multiplication
DEVICE Scalar64 Scalar64_mul_default(Scalar64 a, Scalar64 b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  Scalar64_limb t[Scalar64_LIMBS + 2] = {0};
  printf("vmx: t1: "); t_print(t); printf("\n");
  uchar i = 0;
    Scalar64_limb carry = 0;
    for(uchar j = 0; j < Scalar64_LIMBS; j++)
      t[j] = Scalar64_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    printf("vmx: t2: "); t_print(t); printf("\n");
    t[Scalar64_LIMBS] = Scalar64_add_with_carry(t[Scalar64_LIMBS], &carry);
    printf("vmx: t3: "); t_print(t); printf("\n");
    printf("vmx: carry1: %016lx\n", carry);
    t[Scalar64_LIMBS + 1] = carry;

    carry = 0;
    Scalar64_limb m = Scalar64_INV * t[0];
    printf("vmx: m: %016lx\n", m);
    Scalar64_mac_with_carry(m, Scalar64_P.val[0], t[0], &carry);
    printf("vmx: carry2: %016lx\n", carry);

  Scalar64 result;
  for(uchar i = 0; i < Scalar64_LIMBS; i++) result.val[i] = t[i];

  if(Scalar64_gte(result, Scalar64_P)) result = Scalar64_sub_(result, Scalar64_P);

  return result;
}

KERNEL void test_mul_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
  *result = Scalar64_mul_default(a, b);
}
