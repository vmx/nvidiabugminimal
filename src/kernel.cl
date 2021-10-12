// Defines to make the code work with both, CUDA and OpenCL
#ifdef __NVCC__
  #define DEVICE __device__
  #define GLOBAL
  #define KERNEL extern "C" __global__
  #define LOCAL __shared__
  #define CONSTANT __constant__

  #define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x
  #define GET_GROUP_ID() blockIdx.x
  #define GET_LOCAL_ID() threadIdx.x
  #define GET_LOCAL_SIZE() blockDim.x
  #define BARRIER_LOCAL() __syncthreads()

  typedef unsigned char uchar;

  #define CUDA
#else // OpenCL
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
#endif

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
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
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
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

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

#ifdef CUDA
typedef uint uint32_t;
typedef int  int32_t;
typedef uint limb;

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}


DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

typedef struct {
  int32_t _position;
} chain_t;

DEVICE inline
void chain_init(chain_t *c) {
  c->_position = 0;
}

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=add_cc(a, b);
  else
    r=addc_cc(a, b);
  return r;
}

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madlo_cc(a, b, c);
  else
    r=madloc_cc(a, b, c);
  return r;
}

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madhi_cc(a, b, c);
  else
    r=madhic_cc(a, b, c);
  return r;
}
#endif


#define Scalar32_limb uint
#define Scalar32_LIMBS 8
#define Scalar32_LIMB_BITS 32
#define Scalar32_ONE ((Scalar32){ { 4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881 } })
#define Scalar32_P ((Scalar32){ { 1, 4294967295, 4294859774, 1404937218, 161601541, 859428872, 698187080, 1944954707 } })
#define Scalar32_R2 ((Scalar32){ { 4092763245, 3382307216, 2274516003, 728559051, 1918122383, 97719446, 2673475345, 122214873 } })
#define Scalar32_ZERO ((Scalar32){ { 0, 0, 0, 0, 0, 0, 0, 0 } })
#define Scalar32_INV 4294967295
typedef struct { Scalar32_limb val[Scalar32_LIMBS]; } Scalar32;
#if defined(OPENCL_NVIDIA) || defined(CUDA)

DEVICE Scalar32 Scalar32_sub_nvidia(Scalar32 a, Scalar32 b) {
asm("sub.cc.u32 %0, %0, %8;\r\n"
"subc.cc.u32 %1, %1, %9;\r\n"
"subc.cc.u32 %2, %2, %10;\r\n"
"subc.cc.u32 %3, %3, %11;\r\n"
"subc.cc.u32 %4, %4, %12;\r\n"
"subc.cc.u32 %5, %5, %13;\r\n"
"subc.cc.u32 %6, %6, %14;\r\n"
"subc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
DEVICE Scalar32 Scalar32_add_nvidia(Scalar32 a, Scalar32 b) {
asm("add.cc.u32 %0, %0, %8;\r\n"
"addc.cc.u32 %1, %1, %9;\r\n"
"addc.cc.u32 %2, %2, %10;\r\n"
"addc.cc.u32 %3, %3, %11;\r\n"
"addc.cc.u32 %4, %4, %12;\r\n"
"addc.cc.u32 %5, %5, %13;\r\n"
"addc.cc.u32 %6, %6, %14;\r\n"
"addc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define Scalar32_BITS (Scalar32_LIMBS * Scalar32_LIMB_BITS)
#if Scalar32_LIMB_BITS == 32
  #define Scalar32_mac_with_carry mac_with_carry_32
  #define Scalar32_add_with_carry add_with_carry_32
#elif Scalar32_LIMB_BITS == 64
  #define Scalar32_mac_with_carry mac_with_carry_64
  #define Scalar32_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool Scalar32_gte(Scalar32 a, Scalar32 b) {
  for(char i = Scalar32_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool Scalar32_eq(Scalar32 a, Scalar32 b) {
  for(uchar i = 0; i < Scalar32_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
  #define Scalar32_add_ Scalar32_add_nvidia
  #define Scalar32_sub_ Scalar32_sub_nvidia
#else
  DEVICE Scalar32 Scalar32_add_(Scalar32 a, Scalar32 b) {
    bool carry = 0;
    for(uchar i = 0; i < Scalar32_LIMBS; i++) {
      Scalar32_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  Scalar32 Scalar32_sub_(Scalar32 a, Scalar32 b) {
    bool borrow = 0;
    for(uchar i = 0; i < Scalar32_LIMBS; i++) {
      Scalar32_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE Scalar32 Scalar32_sub(Scalar32 a, Scalar32 b) {
  Scalar32 res = Scalar32_sub_(a, b);
  if(!Scalar32_gte(a, b)) res = Scalar32_add_(res, Scalar32_P);
  return res;
}

// Modular addition
DEVICE Scalar32 Scalar32_add(Scalar32 a, Scalar32 b) {
  Scalar32 res = Scalar32_add_(a, b);
  if(Scalar32_gte(res, Scalar32_P)) res = Scalar32_sub_(res, Scalar32_P);
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void Scalar32_reduce(uint32_t accLow[Scalar32_LIMBS], uint32_t np0, uint32_t fq[Scalar32_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = Scalar32_LIMBS;
  uint32_t accHigh[Scalar32_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void Scalar32_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = Scalar32_LIMBS;
  const uint32_t yLimbs  = Scalar32_LIMBS;
  const uint32_t xyLimbs = Scalar32_LIMBS * 2;
  uint32_t temp[Scalar32_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE Scalar32 Scalar32_mul_nvidia(Scalar32 a, Scalar32 b) {
  // Perform full multiply
  limb ab[2 * Scalar32_LIMBS];
  Scalar32_mult_v1(a.val, b.val, ab);

  uint32_t io[Scalar32_LIMBS];
  #pragma unroll
  for(int i=0;i<Scalar32_LIMBS;i++) {
    io[i]=ab[i];
  }
  Scalar32_reduce(io, Scalar32_INV, Scalar32_P.val);

  // Add io to the upper words of ab
  ab[Scalar32_LIMBS] = add_cc(ab[Scalar32_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < Scalar32_LIMBS - 1; j++) {
    ab[j + Scalar32_LIMBS] = addc_cc(ab[j + Scalar32_LIMBS], io[j]);
  }
  ab[2 * Scalar32_LIMBS - 1] = addc(ab[2 * Scalar32_LIMBS - 1], io[Scalar32_LIMBS - 1]);

  Scalar32 r;
  #pragma unroll
  for (int i = 0; i < Scalar32_LIMBS; i++) {
    r.val[i] = ab[i + Scalar32_LIMBS];
  }

  if (Scalar32_gte(r, Scalar32_P)) {
    r = Scalar32_sub_(r, Scalar32_P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE Scalar32 Scalar32_mul_default(Scalar32 a, Scalar32 b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  Scalar32_limb t[Scalar32_LIMBS + 2] = {0};
  for(uchar i = 0; i < Scalar32_LIMBS; i++) {
    Scalar32_limb carry = 0;
    for(uchar j = 0; j < Scalar32_LIMBS; j++)
      t[j] = Scalar32_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[Scalar32_LIMBS] = Scalar32_add_with_carry(t[Scalar32_LIMBS], &carry);
    t[Scalar32_LIMBS + 1] = carry;

    carry = 0;
    Scalar32_limb m = Scalar32_INV * t[0];
    Scalar32_mac_with_carry(m, Scalar32_P.val[0], t[0], &carry);
    for(uchar j = 1; j < Scalar32_LIMBS; j++)
      t[j - 1] = Scalar32_mac_with_carry(m, Scalar32_P.val[j], t[j], &carry);

    t[Scalar32_LIMBS - 1] = Scalar32_add_with_carry(t[Scalar32_LIMBS], &carry);
    t[Scalar32_LIMBS] = t[Scalar32_LIMBS + 1] + carry;
  }

  Scalar32 result;
  for(uchar i = 0; i < Scalar32_LIMBS; i++) result.val[i] = t[i];

  if(Scalar32_gte(result, Scalar32_P)) result = Scalar32_sub_(result, Scalar32_P);

  return result;
}

#ifdef CUDA
DEVICE Scalar32 Scalar32_mul(Scalar32 a, Scalar32 b) {
  return Scalar32_mul_nvidia(a, b);
}
#else
DEVICE Scalar32 Scalar32_mul(Scalar32 a, Scalar32 b) {
  return Scalar32_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE Scalar32 Scalar32_sqr(Scalar32 a) {
  return Scalar32_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of Scalar32_add(a, a)
DEVICE Scalar32 Scalar32_double(Scalar32 a) {
  for(uchar i = Scalar32_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (Scalar32_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(Scalar32_gte(a, Scalar32_P)) a = Scalar32_sub_(a, Scalar32_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE Scalar32 Scalar32_pow(Scalar32 base, uint exponent) {
  Scalar32 res = Scalar32_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = Scalar32_mul(res, base);
    exponent = exponent >> 1;
    base = Scalar32_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE Scalar32 Scalar32_pow_lookup(GLOBAL Scalar32 *bases, uint exponent) {
  Scalar32 res = Scalar32_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = Scalar32_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE Scalar32 Scalar32_mont(Scalar32 a) {
  return Scalar32_mul(a, Scalar32_R2);
}

DEVICE Scalar32 Scalar32_unmont(Scalar32 a) {
  Scalar32 one = Scalar32_ZERO;
  one.val[0] = 1;
  return Scalar32_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool Scalar32_get_bit(Scalar32 l, uint i) {
  return (l.val[Scalar32_LIMBS - 1 - i / Scalar32_LIMB_BITS] >> (Scalar32_LIMB_BITS - 1 - (i % Scalar32_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint Scalar32_get_bits(Scalar32 l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= Scalar32_get_bit(l, skip + i);
  }
  return ret;
}


#define Scalar64_limb ulong
#define Scalar64_LIMBS 4
#define Scalar64_LIMB_BITS 64
#define Scalar64_ONE ((Scalar64){ { 8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911 } })
#define Scalar64_P ((Scalar64){ { 18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352 } })
#define Scalar64_R2 ((Scalar64){ { 14526898881837571181, 3129137299524312099, 419701826671360399, 524908885293268753 } })
#define Scalar64_ZERO ((Scalar64){ { 0, 0, 0, 0 } })
#define Scalar64_INV 18446744069414584319
typedef struct { Scalar64_limb val[Scalar64_LIMBS]; } Scalar64;
#if defined(OPENCL_NVIDIA) || defined(CUDA)

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

#define Scalar64_BITS (Scalar64_LIMBS * Scalar64_LIMB_BITS)
#if Scalar64_LIMB_BITS == 32
  #define Scalar64_mac_with_carry mac_with_carry_32
  #define Scalar64_add_with_carry add_with_carry_32
#elif Scalar64_LIMB_BITS == 64
  #define Scalar64_mac_with_carry mac_with_carry_64
  #define Scalar64_add_with_carry add_with_carry_64
#endif

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

// Equals
DEVICE bool Scalar64_eq(Scalar64 a, Scalar64 b) {
  for(uchar i = 0; i < Scalar64_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
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


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void Scalar64_reduce(uint32_t accLow[Scalar64_LIMBS], uint32_t np0, uint32_t fq[Scalar64_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = Scalar64_LIMBS;
  uint32_t accHigh[Scalar64_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void Scalar64_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = Scalar64_LIMBS;
  const uint32_t yLimbs  = Scalar64_LIMBS;
  const uint32_t xyLimbs = Scalar64_LIMBS * 2;
  uint32_t temp[Scalar64_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE Scalar64 Scalar64_mul_nvidia(Scalar64 a, Scalar64 b) {
  // Perform full multiply
  limb ab[2 * Scalar64_LIMBS];
  Scalar64_mult_v1(a.val, b.val, ab);

  uint32_t io[Scalar64_LIMBS];
  #pragma unroll
  for(int i=0;i<Scalar64_LIMBS;i++) {
    io[i]=ab[i];
  }
  Scalar64_reduce(io, Scalar64_INV, Scalar64_P.val);

  // Add io to the upper words of ab
  ab[Scalar64_LIMBS] = add_cc(ab[Scalar64_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < Scalar64_LIMBS - 1; j++) {
    ab[j + Scalar64_LIMBS] = addc_cc(ab[j + Scalar64_LIMBS], io[j]);
  }
  ab[2 * Scalar64_LIMBS - 1] = addc(ab[2 * Scalar64_LIMBS - 1], io[Scalar64_LIMBS - 1]);

  Scalar64 r;
  #pragma unroll
  for (int i = 0; i < Scalar64_LIMBS; i++) {
    r.val[i] = ab[i + Scalar64_LIMBS];
  }

  if (Scalar64_gte(r, Scalar64_P)) {
    r = Scalar64_sub_(r, Scalar64_P);
  }

  return r;
}

#endif

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

#ifdef CUDA
DEVICE Scalar64 Scalar64_mul(Scalar64 a, Scalar64 b) {
  return Scalar64_mul_nvidia(a, b);
}
#else
DEVICE Scalar64 Scalar64_mul(Scalar64 a, Scalar64 b) {
  return Scalar64_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE Scalar64 Scalar64_sqr(Scalar64 a) {
  return Scalar64_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of Scalar64_add(a, a)
DEVICE Scalar64 Scalar64_double(Scalar64 a) {
  for(uchar i = Scalar64_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (Scalar64_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(Scalar64_gte(a, Scalar64_P)) a = Scalar64_sub_(a, Scalar64_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE Scalar64 Scalar64_pow(Scalar64 base, uint exponent) {
  Scalar64 res = Scalar64_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = Scalar64_mul(res, base);
    exponent = exponent >> 1;
    base = Scalar64_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE Scalar64 Scalar64_pow_lookup(GLOBAL Scalar64 *bases, uint exponent) {
  Scalar64 res = Scalar64_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = Scalar64_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE Scalar64 Scalar64_mont(Scalar64 a) {
  return Scalar64_mul(a, Scalar64_R2);
}

DEVICE Scalar64 Scalar64_unmont(Scalar64 a) {
  Scalar64 one = Scalar64_ZERO;
  one.val[0] = 1;
  return Scalar64_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool Scalar64_get_bit(Scalar64 l, uint i) {
  return (l.val[Scalar64_LIMBS - 1 - i / Scalar64_LIMB_BITS] >> (Scalar64_LIMB_BITS - 1 - (i % Scalar64_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint Scalar64_get_bits(Scalar64 l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= Scalar64_get_bit(l, skip + i);
  }
  return ret;
}


//KERNEL void test_add_32(Scalar32 a, Scalar32 b, GLOBAL Scalar32 *result) {
//  *result = Scalar32_add(a, b);
//}
//
KERNEL void test_mul_32(Scalar32 a, Scalar32 b, GLOBAL Scalar32 *result) {
  *result = Scalar32_mul(a, b);
}
//
//KERNEL void test_sub_32(Scalar32 a, Scalar32 b, GLOBAL Scalar32 *result) {
//  *result = Scalar32_sub(a, b);
//}
//
//KERNEL void test_pow_32(Scalar32 a, uint b, GLOBAL Scalar32 *result) {
//  *result = Scalar32_pow(a, b);
//}
//
//KERNEL void test_mont_32(Scalar32 a, GLOBAL Scalar32 *result) {
//  *result = Scalar32_mont(a);
//}
//
//KERNEL void test_unmont_32(Scalar32 a, GLOBAL Scalar32 *result) {
//  *result = Scalar32_unmont(a);
//}
//
//KERNEL void test_sqr_32(Scalar32 a, GLOBAL Scalar32 *result) {
//  *result = Scalar32_sqr(a);
//}
//
//KERNEL void test_double_32(Scalar32 a, GLOBAL Scalar32 *result) {
//  *result = Scalar32_double(a);
//}
//
//////////////
//// CUDA doesn't support 64-bit limbs
//#ifndef CUDA
//
//KERNEL void test_add_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
//  *result = Scalar64_add(a, b);
//}
//
//KERNEL void test_mul_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
//  *result = Scalar64_mul(a, b);
//}
//
//KERNEL void test_sub_64(Scalar64 a, Scalar64 b, GLOBAL Scalar64 *result) {
//  *result = Scalar64_sub(a, b);
//}
//
//KERNEL void test_pow_64(Scalar64 a, uint b, GLOBAL Scalar64 *result) {
//  *result = Scalar64_pow(a, b);
//}
//
//KERNEL void test_mont_64(Scalar64 a, GLOBAL Scalar64 *result) {
//  *result = Scalar64_mont(a);
//}
//
//KERNEL void test_unmont_64(Scalar64 a, GLOBAL Scalar64 *result) {
//  *result = Scalar64_unmont(a);
//}
//
//KERNEL void test_sqr_64(Scalar64 a, GLOBAL Scalar64 *result) {
//  *result = Scalar64_sqr(a);
//}
//
//KERNEL void test_double_64(Scalar64 a, GLOBAL Scalar64 *result) {
//  *result = Scalar64_double(a);
//}
//#endif
