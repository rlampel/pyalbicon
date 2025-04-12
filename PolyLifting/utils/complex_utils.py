import casadi as cs


def polar_to_complex(r, phi):
    """Transform polar coordinates into a complex number.

    Keyword arguments:
        r   -- radius
        phi -- angle
    """
    x = r * cs.cos(phi)
    y = r * cs.sin(phi)
    return cs.vertcat(x, y)


def complex_to_polar(comp_num):
    """Transform a complex number into polar coordinates.

    Keyword arguments:
        comp_num -- complex number given as a two-dimensional vector
    """
    r = cs.norm_2(comp_num)
    comp_num /= r
    phi = cs.atan2(comp_num[1], comp_num[0])
    return r, phi


def complex_mult(comp_num_1, comp_num_2):
    """Mulitply two complex numbers

    Keyword arguments:
        comp_num_1, comp_num_2 -- complex numbers given as two-dimensional vectors
    """
    r_1, phi_1 = complex_to_polar(comp_num_1)
    r_2, phi_2 = complex_to_polar(comp_num_2)
    r_res = r_1 * r_2
    phi_res = phi_1 + phi_2
    return polar_to_complex(r_res, phi_res)


def complex_exponent(comp_num, expo):
    """Compute the exponent of a complex number

    Keyword arguments:
        complex_num_1 -- complex number given as a two-dimensional vectors
        expo    -- exponent
    """
    r, phi = complex_to_polar(comp_num)
    r = r**expo
    phi = phi * expo
    return polar_to_complex(r, phi)

