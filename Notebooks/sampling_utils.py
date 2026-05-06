import numpy as np
from linalg_utils import spectral_radius

def sample_C(d_y: int, d_x: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(d_y, d_x))


def sample_L(d_x: int, d_y: int, rng: np.random.Generator) -> np.ndarray:
    # isotropic experiment choice
    return np.eye(d_x, d_y)


def sample_L_anisotropic(
    d_x: int,
    d_y: int,
    rng: np.random.Generator,
    strength_min: float = 0.05,
    strength_max: float = 2.5,
) -> np.ndarray:
    """
    Optional anisotropic alternative for L.
    """
    L = rng.normal(size=(d_x, d_y))
    scales = rng.uniform(strength_min, strength_max, size=d_y)
    return L @ np.diag(scales)


def sample_F_in_band(
    d_x: int,
    rho_min: float,
    rho_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample F ∈ R^{d_x, d_x} and rescale so rho(F) lies in [rho_min, rho_max].
    """
    F_raw = rng.normal(size=(d_x, d_x))
    rho_raw = spectral_radius(F_raw)

    if rho_raw < 1e-12:
        raise RuntimeError("Degenerate F_raw encountered.")

    rho_target = rng.uniform(rho_min, rho_max)
    F = (rho_target / (rho_raw + 1e-12)) * F_raw
    return F


def sample_system_with_F_in_band(
    d_x: int,
    d_y: int,
    rho_min: float,
    rho_max: float,
    rng: np.random.Generator,
    a_rho_max: float = 0.98,
    max_tries_F: int = 200,
    max_tries_CL: int = 200,
    c_scale: float = 1.0,
    l_scale: float = 1.0,
    use_identity_L: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (A, C, L) so that:
      1. F = A - L C has rho(F) in [rho_min, rho_max]
      2. A itself is simulation-stable enough: rho(A) <= a_rho_max

    Improved logic:
      - sample F first and keep it fixed
      - then retry only C, L embeddings for that same F
      - only if all such embeddings fail do we resample a new F

    Construction:
        sample F in-band
        sample C, L
        set A = F + L C
    so that A - L C = F exactly.
    """
    if not (0 <= rho_min <= rho_max < 1):
        raise ValueError(f"Need 0 <= rho_min <= rho_max < 1, got [{rho_min}, {rho_max}]")

    for _ in range(max_tries_F):
        F = sample_F_in_band(d_x, rho_min, rho_max, rng)

        for _ in range(max_tries_CL):
            C = c_scale * sample_C(d_y, d_x, rng)

            if use_identity_L:
                L = l_scale * sample_L(d_x, d_y, rng)
            else:
                L = l_scale * rng.normal(size=(d_x, d_y))

            A = F + L @ C

            rho_A = spectral_radius(A)
            if rho_A <= a_rho_max:
                return A, C, L

    raise RuntimeError(
        f"Could not embed a sampled F with rho(F) in [{rho_min}, {rho_max}] "
        f"into a system satisfying rho(A) <= {a_rho_max} "
        f"after {max_tries_F * max_tries_CL} total tries."
    )