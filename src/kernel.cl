  #define DEVICE
  #define GLOBAL __global
  #define KERNEL __kernel
  #define LOCAL __local
  #define CONSTANT __constant

  #define GET_GLOBAL_ID() get_global_id(0)
  #define GET_GROUP_ID() get_group_id(0)
  #define GET_LOCAL_ID() get_local_id(0)
  #define GET_LOCAL_SIZE() get_local_size(0)
  #define BARRIER_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #if defined(OPENCL_NVIDIA)
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #if defined(OPENCL_NVIDIA)
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

#define Scalar64_limb ulong
#define Scalar64_LIMBS 4
#define Scalar64_P ((Scalar64){ { 18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352 } })
#define Scalar64_INV 18446744069414584319
typedef struct { Scalar64_limb val[Scalar64_LIMBS]; } Scalar64;
#if defined(OPENCL_NVIDIA)

DEVICE Scalar64 Scalar64_sub_nvidia(Scalar64 a, Scalar64 b) {
asm("sub.cc.u64 %0, %0, %4;\r\n"
"subc.cc.u64 %1, %1, %5;\r\n"
"subc.cc.u64 %2, %2, %6;\r\n"
"subc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
DEVICE Scalar64 Scalar64_add_nvidia(Scalar64 a, Scalar64 b) {
asm("add.cc.u64 %0, %0, %4;\r\n"
"addc.cc.u64 %1, %1, %5;\r\n"
"addc.cc.u64 %2, %2, %6;\r\n"
"addc.u64 %3, %3, %7;\r\n"
:"+l"(a.val[0]), "+l"(a.val[1]), "+l"(a.val[2]), "+l"(a.val[3])
:"l"(b.val[0]), "l"(b.val[1]), "l"(b.val[2]), "l"(b.val[3]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

  #define Scalar64_mac_with_carry mac_with_carry_64
  #define Scalar64_add_with_carry add_with_carry_64

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
#if defined(OPENCL_NVIDIA)
  #define Scalar64_add_ Scalar64_add_nvidia
  #define Scalar64_sub_ Scalar64_sub_nvidia
#else
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
#endif

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
  for(uchar i = 0; i < Scalar64_LIMBS; i++) {
    Scalar64_limb carry = 0;
    for(uchar j = 0; j < Scalar64_LIMBS; j++)
      t[j] = Scalar64_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[Scalar64_LIMBS] = Scalar64_add_with_carry(t[Scalar64_LIMBS], &carry);
    t[Scalar64_LIMBS + 1] = carry;

    carry = 0;
    Scalar64_limb m = Scalar64_INV * t[0];
    Scalar64_mac_with_carry(m, Scalar64_P.val[0], t[0], &carry);
    for(uchar j = 1; j < Scalar64_LIMBS; j++)
      t[j - 1] = Scalar64_mac_with_carry(m, Scalar64_P.val[j], t[j], &carry);

    t[Scalar64_LIMBS - 1] = Scalar64_add_with_carry(t[Scalar64_LIMBS], &carry);
    t[Scalar64_LIMBS] = t[Scalar64_LIMBS + 1] + carry;
  }

  Scalar64 result;
  for(uchar i = 0; i < Scalar64_LIMBS; i++) result.val[i] = t[i];

  if(Scalar64_gte(result, Scalar64_P)) result = Scalar64_sub_(result, Scalar64_P);

  return result;
}

KERNEL void test_mul_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
  *result = Scalar64_mul_default(a, b);
}
