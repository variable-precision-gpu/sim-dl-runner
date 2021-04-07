import math, struct

def mpfr_exponent_range(exponent_width, significand_width):
  assert(exponent_width >= 1)
  assert(significand_width >= 1)

  # exponent range
  exponent_range = 2**exponent_width
  # account for IEEE special numbers
  exponent_range -= 2
  # max and min exponent with IEEE style offset
  maximum = exponent_range / 2 
  minimum = - exponent_range / 2 + 1
  # account for MPFR significand representation
  mpfr_offset = 1
  maximum += mpfr_offset
  minimum += mpfr_offset
  # account for subnormal exponent range
  minimum -= significand_width

  print("VF32_SIGNIFICAND  = {}".format(significand_width + mpfr_offset))
  print("VF32_EXPONENT_MIN = {}".format(int(minimum)))
  print("VF32_EXPONENT_MAX = {}".format(int(maximum)))
  return minimum, maximum

def exponent_no_subnormal_range(exponent_width):
  assert(exponent_width > 1)

  range = 2**exponent_width
  maximum = range / 2 - 1
  minimum = - range / 2 + 2
  return minimum, maximum

def components(value):
  significand, exponent = math.frexp(value)
  exponent = exponent - 1
  significand = significand * 2
  return exponent, significand

def significand_length(value):
  # from: https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
  bit_representation = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))
  significand = len(bit_representation[9:].rstrip('0'))
  return significand
  
