import math
import struct


def mpfr_exponent_range(exponent_width, significand_width):
    """Computes the appropriate MPFR exponent range to set to emulate a given IEEE 754-style type.

    Accounts for the subnormal range granted by the significand width, required to emulate IEEE 754
    types in MPFR. This function also prints the appropriate environment variables to set in the simulator,
    for convenience.

    Args:
        exponent_width (int): number of bits in exponent.
        significand_width (int): number of explicit bits in significand.

    Returns:
        (int, int): minimum and maximum exponent that should be set in MPFR to emulate the component widths.
    """
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

    print("VF_SIGNIFICAND  = {}".format(significand_width + mpfr_offset))
    print("VF_EXPONENT_MIN = {}".format(int(minimum)))
    print("VF_EXPONENT_MAX = {}".format(int(maximum)))
    return minimum, maximum


def exponent_no_subnormal_range(exponent_width):
    """Computes the appropriate IEEE 754 exponent range an exponent width.

    Does not account for subnormal range, which is dependent on the significand width.

    Args:
        exponent_width (int): number of bits in exponent.

    Returns:
        (int, int): minimum and maximum exponent granted by an exponent width.
    """
    assert(exponent_width > 1)

    range = 2**exponent_width
    maximum = range / 2 - 1
    minimum = - range / 2 + 2
    return minimum, maximum


def components(value):
    """Computes the exponent and significand components of a number.

    Args:
        value (float): input number.

    Returns:
        (int, float): exponent and significand components that make up the input value.
    """
    significand, exponent = math.frexp(value)
    exponent = exponent - 1
    significand = significand * 2
    return exponent, significand


def significand_length(value):
    """Computes the IEEE 754 significand length of a value.

    From: https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex

    Args:
        value (float): input number.

    Returns:
        (int): the significand bit length.
    """
    bit_representation = ''.join(bin(c).replace('0b', '').rjust(
        8, '0') for c in struct.pack('!f', value))
    significand = len(bit_representation[9:].rstrip('0'))
    return significand
