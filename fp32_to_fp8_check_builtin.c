// fp32_to_fp8_check_builtin.c
// Convert IEEE-754 FP32 -> FP8 stored in uint8_t in two ways:
//   1. Reference C implementation (fp32_to_fp8_ref)
//   2. Compiler builtin (__builtin_convert_to_arbitrary_fp)
//
// Then compare results and report mismatches.
//
// Formats tested:
//   Float8E5M2   (IEEE754: has Inf and NaN)
//   Float8E4M3FN (NanOnly + AllOnes NaN encoding, no Inf)
//
// Build (clang fork with the builtin):
//   clang -O2 -o fp32_to_fp8_check fp32_to_fp8_check_builtin.c -lm
//
// This is a runtime test for
// https://github.com/MrSidims/llvm-project/tree/expand-apfloat-to-apfloat

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Encoding: 0 = Float8E5M2, 1 = Float8E4M3FN
// Rounding: 0 = NearestTiesToEven
#define FP32_TO_FP8_E5M2(val)                                                  \
  __builtin_convert_to_arbitrary_fp((val), 0, 0, 0)
#define FP32_TO_FP8_E4M3FN(val)                                                \
  __builtin_convert_to_arbitrary_fp((val), 1, 0, 0)
#define FP32_TO_FP8_E5M2_SAT(val)                                              \
  __builtin_convert_to_arbitrary_fp((val), 0, 0, 1)
#define FP32_TO_FP8_E4M3FN_SAT(val)                                            \
  __builtin_convert_to_arbitrary_fp((val), 1, 0, 1)

typedef enum {
  FP8_E5M2 = 0,
  FP8_E4M3FN = 1,
} Fp8Format;

typedef struct {
  const char *Name;
  int ExpBits;
  int MantBits;
  int Bias;
  int MaxExpField;
  uint8_t MantMask;
  int HasInfinity;
  int NanIsAllOnes;
  // Max finite: exp field and mant field for the largest finite value.
  int MaxFiniteExp;
  int MaxFiniteMant;
} Fp8Spec;

static Fp8Spec get_fp8_spec(Fp8Format Format) {
  Fp8Spec S;
  if (Format == FP8_E4M3FN) {
    S.Name = "E4M3FN";
    S.ExpBits = 4;
    S.MantBits = 3;
    S.Bias = 7;
    S.MaxExpField = 15;
    S.MantMask = 0x07;
    S.HasInfinity = 0;
    S.NanIsAllOnes = 1;
    S.MaxFiniteExp = 15;
    S.MaxFiniteMant = 6; // 0x7E = 0_1111_110, NaN is 0x7F = 0_1111_111
  } else {
    S.Name = "E5M2";
    S.ExpBits = 5;
    S.MantBits = 2;
    S.Bias = 15;
    S.MaxExpField = 31;
    S.MantMask = 0x03;
    S.HasInfinity = 1;
    S.NanIsAllOnes = 0;
    S.MaxFiniteExp = 30;
    S.MaxFiniteMant = 3; // 0x7B = 0_11110_11
  }
  return S;
}

static uint32_t float_bits(float V) {
  uint32_t B;
  memcpy(&B, &V, sizeof(B));
  return B;
}

static float bits_to_float(uint32_t B) {
  float V;
  memcpy(&V, &B, sizeof(V));
  return V;
}

// Reference: convert f32 -> fp8 with round-to-nearest-even.
static uint8_t fp32_to_fp8_ref(float Val, Fp8Format Format) {
  const Fp8Spec S = get_fp8_spec(Format);
  const uint32_t Bits = float_bits(Val);
  const int Sign = (Bits >> 31) & 1;
  const int SrcExp = (int)((Bits >> 23) & 0xFF);
  const uint32_t SrcMant = Bits & 0x7FFFFF;

  // NaN
  if (SrcExp == 255 && SrcMant != 0) {
    if (S.HasInfinity) {
      // IEEE754 E5M2: canonical qNaN with quiet bit set.
      // qNaN bit = 1 << (MantBits - 1)
      uint8_t QNaN = (uint8_t)((S.MaxExpField << S.MantBits) |
                                (1 << (S.MantBits - 1)));
      return QNaN;
    } else if (S.NanIsAllOnes) {
      // E4M3FN: NaN = 0_1111_111 = 0x7F
      return 0x7F;
    }
    // FiniteOnly: shouldn't happen for these formats, return 0
    return 0;
  }

  // Inf
  if (SrcExp == 255 && SrcMant == 0) {
    if (S.HasInfinity) {
      // E5M2: signed infinity
      return (uint8_t)((Sign << 7) | (S.MaxExpField << S.MantBits));
    } else {
      // No Inf and not saturating: result is poison/undefined.
      // Return 0xFF as sentinel (caller should not test this case).
      return 0xFF;
    }
  }

  // Zero
  if (SrcExp == 0 && SrcMant == 0)
    return (uint8_t)(Sign << 7);

  // Compute true (unbiased) exponent and full mantissa with implicit 1.
  int TrueExp;
  uint32_t FullMant; // 24-bit: bit 23 is implicit 1
  if (SrcExp == 0) {
    // Source denormal: true_exp = 1 - 127 = -126
    TrueExp = -126;
    FullMant = SrcMant; // no implicit 1
    // Normalize
    while (FullMant != 0 && (FullMant & (1u << 23)) == 0) {
      FullMant <<= 1;
      TrueExp--;
    }
  } else {
    TrueExp = SrcExp - 127;
    FullMant = (1u << 23) | SrcMant;
  }

  // Now FullMant has bit 23 as the leading 1.
  // Compute biased exponent in destination.
  int DstBiasedExp = TrueExp + S.Bias;

  // Overflow check.
  if (DstBiasedExp > S.MaxFiniteExp) {
    if (S.HasInfinity)
      return (uint8_t)((Sign << 7) | (S.MaxExpField << S.MantBits));
    else
      return 0xFF; // poison/undefined without saturation
  }

  // Normal destination.
  if (DstBiasedExp >= 1) {
    // Shift mantissa: we have 23 fractional bits, need MantBits.
    unsigned Shift = 23 - S.MantBits;
    uint32_t TruncMant = (FullMant >> Shift) & S.MantMask;

    // Rounding (round-to-nearest-even).
    uint32_t RoundBit = (FullMant >> (Shift - 1)) & 1;
    uint32_t StickyBits = FullMant & ((1u << (Shift - 1)) - 1);
    uint32_t LSB = TruncMant & 1;

    if (RoundBit && (StickyBits || LSB))
      TruncMant++;

    // Mantissa overflow from rounding.
    if (TruncMant > (uint32_t)S.MantMask) {
      TruncMant = 0;
      DstBiasedExp++;
    }

    // Check exp overflow after rounding.
    if (DstBiasedExp > S.MaxFiniteExp) {
      if (S.HasInfinity)
        return (uint8_t)((Sign << 7) | (S.MaxExpField << S.MantBits));
      else
        return 0xFF; // poison/undefined without saturation
    }

    // Check NaN encoding collision for NanOnly formats.
    if (S.NanIsAllOnes && DstBiasedExp == S.MaxExpField &&
        (int)TruncMant > S.MaxFiniteMant) {
      TruncMant = S.MaxFiniteMant;
    }

    return (uint8_t)((Sign << 7) | (DstBiasedExp << S.MantBits) |
                      (TruncMant & S.MantMask));
  }

  // Destination denormal: DstBiasedExp <= 0.
  // denorm_shift = 1 - DstBiasedExp
  int DenormShift = 1 - DstBiasedExp;
  // Total shift from 23-bit mantissa: 23 - MantBits + DenormShift
  unsigned TotalShift = (unsigned)(23 - S.MantBits + DenormShift);

  if (TotalShift >= 32) {
    // Too small, round to zero or smallest denorm depending on rounding.
    return (uint8_t)(Sign << 7);
  }

  uint32_t TruncMant = FullMant >> TotalShift;

  // Rounding.
  if (TotalShift >= 1) {
    uint32_t RoundBit = (FullMant >> (TotalShift - 1)) & 1;
    uint32_t StickyBits =
        (TotalShift >= 2) ? (FullMant & ((1u << (TotalShift - 1)) - 1)) : 0;
    uint32_t LSB = TruncMant & 1;

    if (RoundBit && (StickyBits || LSB))
      TruncMant++;
  }

  // Denorm rounding overflow -> smallest normal.
  if (TruncMant > (uint32_t)S.MantMask)
    return (uint8_t)((Sign << 7) | (1 << S.MantBits));

  return (uint8_t)((Sign << 7) | (TruncMant & S.MantMask));
}

// Builtin conversion.
static uint8_t fp32_to_fp8_builtin(float Val, Fp8Format Format) {
  if (Format == FP8_E4M3FN)
    return (uint8_t)FP32_TO_FP8_E4M3FN(Val);
  else
    return (uint8_t)FP32_TO_FP8_E5M2(Val);
}

static void print_mismatch(const Fp8Spec *S, float Val, uint8_t Ref,
                           uint8_t Builtin) {
  printf("MISMATCH %s f32=0x%08X (%a)\n", S->Name, float_bits(Val), Val);
  printf("  ref     : 0x%02X\n", (unsigned)Ref);
  printf("  builtin : 0x%02X\n", (unsigned)Builtin);
}

static int check_value(Fp8Format Format, float Val, int Verbose) {
  const Fp8Spec S = get_fp8_spec(Format);

  uint8_t Ref = fp32_to_fp8_ref(Val, Format);
  // 0xFF sentinel means the result is poison/undefined (e.g. overflow in a
  // format without Inf and without saturation). Skip the comparison.
  if (Ref == 0xFF) {
    if (Verbose)
      printf("SKIP %s f32=%a -> undefined (no Inf, no saturation)\n", S.Name,
             Val);
    return 0;
  }

  uint8_t Builtin = fp32_to_fp8_builtin(Val, Format);

  int Match = (Ref == Builtin);
  if (!Match) {
    print_mismatch(&S, Val, Ref, Builtin);
  } else if (Verbose) {
    printf("OK %s f32=%a -> 0x%02X\n", S.Name, Val, (unsigned)Ref);
  }
  return Match ? 0 : 1;
}

// Convert a fp8 raw byte back to float for roundtrip testing.
static float fp8_to_float_ref(uint8_t Raw, Fp8Format Format) {
  const Fp8Spec S = get_fp8_spec(Format);
  int Sign = (Raw >> 7) & 1;
  int ExpField = (Raw >> S.MantBits) & ((1 << S.ExpBits) - 1);
  int MantField = Raw & S.MantMask;

  if (S.NanIsAllOnes && ExpField == S.MaxExpField &&
      MantField == S.MantMask + 1 - 1)
    if (ExpField == S.MaxExpField && MantField == (int)S.MantMask)
      return NAN;
  if (!S.NanIsAllOnes && ExpField == S.MaxExpField && MantField != 0)
    return NAN;
  if (S.HasInfinity && ExpField == S.MaxExpField && MantField == 0)
    return Sign ? -INFINITY : INFINITY;
  if (ExpField == 0 && MantField == 0)
    return Sign ? -0.0f : 0.0f;

  float Frac;
  int UnbiasedExp;
  if (ExpField == 0) {
    Frac = (float)MantField / (float)(1 << S.MantBits);
    UnbiasedExp = 1 - S.Bias;
  } else {
    Frac = 1.0f + (float)MantField / (float)(1 << S.MantBits);
    UnbiasedExp = ExpField - S.Bias;
  }
  float Val = ldexpf(Frac, UnbiasedExp);
  return Sign ? -Val : Val;
}

// Test: for every representable fp8 value, convert to f32 and back.
// The roundtrip should be exact (the fp8 value is representable in f32).
static int run_roundtrip_test(Fp8Format Format) {
  const Fp8Spec S = get_fp8_spec(Format);
  printf("\n=== Roundtrip test for %s ===\n", S.Name);

  int Failures = 0;
  for (unsigned Raw = 0; Raw < 256; ++Raw) {
    uint8_t Raw8 = (uint8_t)Raw;
    float F = fp8_to_float_ref(Raw8, Format);

    // Skip NaN (roundtrip produces canonical NaN, not necessarily same bits)
    if (isnan(F))
      continue;

    Failures += check_value(Format, F, /*Verbose=*/0);
  }

  if (Failures == 0)
    printf("All roundtrip values matched for %s.\n", S.Name);
  else
    printf("Failures for %s: %d mismatches.\n", S.Name, Failures);

  return Failures;
}

// Test specific edge cases.
static int run_edge_case_tests(Fp8Format Format) {
  const Fp8Spec S = get_fp8_spec(Format);
  printf("\n=== Edge cases for %s ===\n", S.Name);

  int Failures = 0;

  // Zero
  Failures += check_value(Format, 0.0f, 1);
  Failures += check_value(Format, -0.0f, 1);

  // NaN
  Failures += check_value(Format, NAN, 1);

  // Inf (for IEEE754 format)
  Failures += check_value(Format, INFINITY, 1);
  Failures += check_value(Format, -INFINITY, 1);

  // 1.0
  Failures += check_value(Format, 1.0f, 1);
  Failures += check_value(Format, -1.0f, 1);

  // Very large value (overflow)
  Failures += check_value(Format, 1.0e10f, 1);
  Failures += check_value(Format, -1.0e10f, 1);

  // Very small value (denorm territory)
  Failures += check_value(Format, 1.0e-10f, 1);
  Failures += check_value(Format, -1.0e-10f, 1);

  // Values near rounding boundaries
  Failures += check_value(Format, 1.5f, 1);
  Failures += check_value(Format, 2.5f, 1);
  Failures += check_value(Format, 0.5f, 1);
  Failures += check_value(Format, 0.25f, 1);

  return Failures;
}

// Test with runtime-unknown values to prevent constant folding.
static int run_runtime_test(void) {
  printf("\n=== Runtime-unknown value tests ===\n");

  int Failures = 0;
  // Use volatile to prevent constant folding.
  volatile float Vals[] = {0.0f,    -0.0f,    1.0f,      -1.0f,   0.5f,
                           2.0f,    100.0f,   -100.0f,   1.0e-6f, 1.0e6f,
                           INFINITY, -INFINITY, NAN};
  int N = sizeof(Vals) / sizeof(Vals[0]);

  for (int I = 0; I < N; ++I) {
    float V = Vals[I];
    Failures += check_value(FP8_E5M2, V, 1);
    Failures += check_value(FP8_E4M3FN, V, 1);
  }

  return Failures;
}

// Saturation test: Inf and overflow should saturate to max finite.
static int check_sat_value(Fp8Format Format, float Val, uint8_t Expected,
                           int Verbose) {
  const Fp8Spec S = get_fp8_spec(Format);
  uint8_t Builtin;
  if (Format == FP8_E4M3FN)
    Builtin = (uint8_t)FP32_TO_FP8_E4M3FN_SAT(Val);
  else
    Builtin = (uint8_t)FP32_TO_FP8_E5M2_SAT(Val);

  int Match = (Expected == Builtin);
  if (!Match) {
    printf("MISMATCH %s SAT f32=%a  expected=0x%02X  got=0x%02X\n", S.Name,
           Val, (unsigned)Expected, (unsigned)Builtin);
  } else if (Verbose) {
    printf("OK %s SAT f32=%a -> 0x%02X\n", S.Name, Val, (unsigned)Builtin);
  }
  return Match ? 0 : 1;
}

static int run_saturation_tests(void) {
  printf("\n=== Saturation tests ===\n");
  int Failures = 0;

  // E5M2: max finite = 0x7B = 0_11110_11 = 57344.0
  // Overflow with saturation -> max finite
  Failures += check_sat_value(FP8_E5M2, 100000.0f, 0x7B, 1);
  Failures += check_sat_value(FP8_E5M2, -100000.0f, 0xFB, 1);
  // E5M2 is IEEE754: Inf is representable, so Inf stays as Inf even with sat.
  Failures += check_sat_value(FP8_E5M2, INFINITY, 0x7C, 1);
  Failures += check_sat_value(FP8_E5M2, -INFINITY, 0xFC, 1);

  // E4M3FN: max finite = 0x7E = 0_1111_110 = 448.0
  // Overflow with saturation -> max finite
  Failures += check_sat_value(FP8_E4M3FN, 1000.0f, 0x7E, 1);
  Failures += check_sat_value(FP8_E4M3FN, -1000.0f, 0xFE, 1);
  // Inf with saturation -> max finite
  Failures += check_sat_value(FP8_E4M3FN, INFINITY, 0x7E, 1);
  Failures += check_sat_value(FP8_E4M3FN, -INFINITY, 0xFE, 1);

  return Failures;
}

int main(void) {
  int TotalFailures = 0;

  TotalFailures += run_edge_case_tests(FP8_E5M2);
  TotalFailures += run_edge_case_tests(FP8_E4M3FN);

  TotalFailures += run_roundtrip_test(FP8_E5M2);
  TotalFailures += run_roundtrip_test(FP8_E4M3FN);

  TotalFailures += run_runtime_test();

  TotalFailures += run_saturation_tests();

  if (TotalFailures == 0) {
    printf("\nSUCCESS: reference matches builtin for all tested values.\n");
    return 0;
  }

  printf("\nFAIL: total mismatches = %d\n", TotalFailures);
  return 1;
}
