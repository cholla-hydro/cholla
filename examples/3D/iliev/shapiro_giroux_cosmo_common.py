"""
shapiro_giroux_cosmo_common.py

Shared physics and I/O helpers for the Shapiro_Giroux_Cosmo analysis scripts.

CONVENTIONS (verified against the cosmo-rt branch)
--------------------------------------------------
* Box coordinates written to the HDF5 file are COMOVING kpc/h
  (chemistry_functions.cpp: physical cm per code length = kpc/h * a).
* The `density` (and species density) datasets are written after
  Change_Cosmological_Frame_System(false), i.e. code density * rho_0_gas.
  That is a COMOVING mass density in units of  h^2 Msun / kpc^3.
      comoving rho [g/cm^3]  =  rho_out * density_unit * h^2
      physical rho           =  comoving rho / a^3
  (same conversion the chemistry uses: dens * density_conv / a^3).

ANALYTIC SOLUTION (Shapiro & Giroux 1987, ApJ 321, L107)
--------------------------------------------------------
The paper normalizes the scale factor to 1 at source turn-on, and uses lengths
that equal proper lengths at turn-on.  Writing a for the usual scale factor
(a = 1 today), a_i = 1/(1+z_i), and n_c for the comoving H number density:

    K       = [ 3 Ndot / (4 pi alpha_B n_c^2) ]^(1/3)      (a length)
    R_S(a)  = a * K                  comoving Stromgren radius       (eq. 3)
    lambda  = alpha_B * n_H,i * t_i,   n_H,i = n_c / a_i^3           (after eq. 5)
    y(a)    = (r_I / r_S,i)^3 solves  dy/dx = lambda (1 - y / a_p^3)  (eq. 5)
              with x = t/t_i and a_p = a/a_i
    R_I(a)  = a_i * K * y^(1/3)      comoving ionization-front radius

For an Einstein-de Sitter universe y has the closed form of eq. (10a).  The
ODE is integrated numerically for any cosmology, and the EdS closed form is
used as a cross-check.

Note: the analytic curve uses alpha_B = 2.6e-13 (T/1e4 K)^-0.8 as in S&G.
Cholla's chemistry uses its own recombination-rate table, so percent-level
differences are expected.
"""

import math

import h5py
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import expn  # expn(2, x) = E_2(x)

# ---------------------------------------------------------------------------
# Constants (CGS), matching src/global/global.h where applicable
# ---------------------------------------------------------------------------
MP = 1.672622e-24  # proton mass [g]
G_CGS = 6.674e-8  # [cm^3 g^-1 s^-2]
MPC_CM = 3.0857e24  # [cm]
KPC_CM = 3.08567758e21  # [cm]  (Cholla LENGTH_UNIT)
MYR_S = 31.536e12  # [s]
NDOT = 5.0e48  # source luminosity [photons/s] (set in Shapiro_Giroux_Cosmo)
ALPHA_B_1E4 = 2.6e-13  # Case B recombination coefficient at 1e4 K [cm^3/s]
MASS_UNIT_FALLBACK = 1.98847e33  # MASS_UNIT in src/global/global.h


# ---------------------------------------------------------------------------
# Parameter file
# ---------------------------------------------------------------------------
def parse_params(param_file):
    """Parse a Cholla parameter file into a dict of typed values."""
    params = {}
    with open(param_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.split("#")[0].strip().strip('"')
            try:
                params[key] = int(val)
            except ValueError:
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    return params


# ---------------------------------------------------------------------------
# Cosmology and analytic solution
# ---------------------------------------------------------------------------
def alpha_B(T):
    """Case B recombination coefficient [cm^3/s]."""
    return ALPHA_B_1E4 * (T / 1.0e4) ** (-0.8)


def hubble(a, c):
    """H(a) in 1/s."""
    return c["H0_cgs"] * math.sqrt(
        c["Omega_R"] / a**4 + c["Omega_M"] / a**3 + c["Omega_K"] / a**2 + c["Omega_L"]
    )


def cosmic_time(a, c):
    """Cosmic time [s] at scale factor a:  t = int_0^a da' / (a' H(a'))."""
    val, _ = quad(
        lambda ap: 1.0 / (ap * hubble(ap, c)), 1.0e-10, a, epsabs=0.0, epsrel=1e-10
    )
    return val


def derive_cosmology(P, ndot=NDOT):
    """Derive every quantity the analytic solution needs from the parameter file."""
    H0 = float(P["H0"])
    h = H0 / 100.0
    Omega_M = float(P["Omega_M"])
    Omega_L = float(P.get("Omega_L", 0.0))
    Omega_R = float(P.get("Omega_R", 0.0))
    c = dict(
        H0=H0,
        h=h,
        Omega_M=Omega_M,
        Omega_L=Omega_L,
        Omega_R=Omega_R,
        Omega_K=1.0 - Omega_M - Omega_L - Omega_R,
        Omega_b=float(P["Omega_b"]),
        YHe=float(P.get("YHe", 0.0)),
        T_init=float(P.get("T_init", 1.0e4)),
        z_i=float(P["Init_redshift"]),
        ndot=ndot,
    )
    c["a_i"] = 1.0 / (1.0 + c["z_i"])
    c["H0_cgs"] = H0 * 1.0e5 / MPC_CM
    c["q_0"] = 0.5 * Omega_M - Omega_L

    rho_crit = 3.0 * c["H0_cgs"] ** 2 / (8.0 * math.pi * G_CGS)
    c["rho_crit"] = rho_crit
    c["nH_comov"] = (1.0 - c["YHe"]) * c["Omega_b"] * rho_crit / MP  # cm^-3
    c["nH_i"] = c["nH_comov"] / c["a_i"] ** 3  # cm^-3, proper at z_i
    c["alpha_B"] = alpha_B(c["T_init"])
    c["t_i"] = cosmic_time(c["a_i"], c)  # s
    c["lam"] = c["alpha_B"] * c["nH_i"] * c["t_i"]
    c["K_kpc"] = (3.0 * ndot / (4.0 * math.pi * c["alpha_B"] * c["nH_comov"] ** 2)) ** (
        1.0 / 3.0
    ) / KPC_CM
    return c


def y_front(a, c):
    """y(a) = (r_I/r_S,i)^3 from numerical integration of S&G eq. (5)."""
    if a <= c["a_i"]:
        return 0.0

    def rhs(ap, y):
        return [
            c["lam"]
            * (1.0 - y[0] * (c["a_i"] / ap) ** 3)
            / (c["t_i"] * ap * hubble(ap, c))
        ]

    sol = solve_ivp(rhs, (c["a_i"], a), [0.0], method="LSODA", rtol=1e-10, atol=1e-14)
    return float(sol.y[0, -1])


def y_eds_closed_form(t, c):
    """S&G eq. (10a); valid only for Einstein-de Sitter (q_0 = 1/2)."""
    lam, ti = c["lam"], c["t_i"]
    r = ti / t
    return lam * math.exp(lam * r) * ((t / ti) * expn(2, lam * r) - expn(2, lam))


def expected_radii(a, c):
    """
    Expected radii at scale factor a, in COMOVING kpc (a = 1 today).
    Returns dict with R_S, R_I, y, t [s], and (for EdS) the closed-form y.
    """
    y = y_front(a, c)
    out = dict(
        y=y,
        t=cosmic_time(a, c),
        R_S=a * c["K_kpc"],
        R_I=c["a_i"] * c["K_kpc"] * max(y, 0.0) ** (1.0 / 3.0),
        y_closed=None,
    )
    if (
        abs(c["q_0"] - 0.5) < 1e-6
        and abs(c["Omega_L"]) < 1e-12
        and abs(c["Omega_R"]) < 1e-12
    ):
        out["y_closed"] = y_eds_closed_form(out["t"], c)
    return out


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
LENGTH_UNIT_CHOICES = ("ckpc/h", "ckpc", "pkpc")


def native_length_factor(units, h, a):
    """Factor converting native coordinates (comoving kpc/h) to `units`."""
    return {"ckpc/h": 1.0, "ckpc": 1.0 / h, "pkpc": a / h}[units]


def ckpc_to_native(h):
    """Factor converting comoving kpc to native comoving kpc/h."""
    return h


def length_label(units):
    return {
        "ckpc/h": r"comoving kpc/$h$",
        "ckpc": "comoving kpc",
        "pkpc": "proper kpc",
    }[units]


def density_to_nH(rho_out, attrs, c, a):
    """
    Convert an output density array (h^2 Msun/kpc^3, comoving) to hydrogen
    number densities [cm^-3].  Returns (n_comoving, n_physical).
    """
    density_unit = float(attrs.get("density_unit", MASS_UNIT_FALLBACK / KPC_CM**3))
    rho_comov_cgs = np.asarray(rho_out, dtype=np.float64) * density_unit * c["h"] ** 2
    n_comov = (1.0 - c["YHe"]) * rho_comov_cgs / MP
    return n_comov, n_comov / a**3


def species_number_density(rho_species_out, attrs, c, a):
    """
    Convert an output species mass density (e.g. HI_density) to a number density [cm^-3].
    Species densities are written in the same frame as `density` (comoving, h^2 Msun/kpc^3),
    but they are already hydrogen mass densities, so no (1 - YHe) factor is applied.
    Returns (n_comoving, n_physical).
    """
    density_unit = float(attrs.get("density_unit", MASS_UNIT_FALLBACK / KPC_CM**3))
    n_comov = (
        np.asarray(rho_species_out, dtype=np.float64) * density_unit * c["h"] ** 2 / MP
    )
    return n_comov, n_comov / a**3


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------
def read_attrs(hdf5_file):
    """Return the root attributes of a Cholla output file as plain Python/numpy."""
    with h5py.File(hdf5_file, "r") as f:
        attrs = {}
        for k, v in f.attrs.items():
            if isinstance(v, bytes):
                v = v.decode()
            attrs[k] = v
    return attrs


def read_field(hdf5_file, name):
    """Read one 3-D dataset, indexed [ix, iy, iz]."""
    with h5py.File(hdf5_file, "r") as f:
        if name not in f:
            raise KeyError(
                f"dataset '{name}' not in {hdf5_file}; available: {list(f.keys())}"
            )
        return f[name][()]


def box_geometry(attrs, P):
    """Return (bounds[3], domain[3], dx[3]) in native units (comoving kpc/h)."""
    if "nprocs" in attrs and int(np.prod(attrs["nprocs"])) > 1:
        raise SystemExit(
            "This script reads a single-process output file "
            f"(file has nprocs = {tuple(attrs['nprocs'])}); concatenate first."
        )
    bounds = np.array(
        attrs.get(
            "bounds", [P.get("xmin", 0.0), P.get("ymin", 0.0), P.get("zmin", 0.0)]
        ),
        float,
    )
    domain = np.array(attrs.get("domain", [P["xlen"], P["ylen"], P["zlen"]]), float)
    dx = np.array(
        attrs.get(
            "dx", [domain[0] / P["nx"], domain[1] / P["ny"], domain[2] / P["nz"]]
        ),
        float,
    )
    return bounds, domain, dx


# ---------------------------------------------------------------------------
# Temperature (requires the 'temperature' dataset, i.e. -DOUTPUT_TEMPERATURE)
# ---------------------------------------------------------------------------
TEMPERATURE_HINT = (
    "no 'temperature' dataset in the file: build with -DOUTPUT_TEMPERATURE "
    "(config/make.type.cosmo-m1 or config/make.type.m1)"
)
NEUTRAL_XHII = 0.1  # cells with x_HII below this count as neutral
IONIZED_XHII = 0.9  # cells with x_HII above this count as ionized
T_IONIZED_RANGE = (3.0e3, 1.0e5)  # plausible range for photoheated gas [K]


def read_optional_field(hdf5_file, name):
    """Read one 3-D dataset, or return None if the file does not contain it."""
    with h5py.File(hdf5_file, "r") as f:
        return f[name][()] if name in f else None


def ionized_fraction(hdf5_file):
    """x_HII = HII_density / (HI_density + HII_density); unit-free."""
    hii = read_field(hdf5_file, "HII_density").astype(np.float64)
    hi = read_field(hdf5_file, "HI_density").astype(np.float64)
    tot = hi + hii
    return np.divide(hii, tot, out=np.zeros_like(tot), where=tot > 0)


def expected_neutral_temperature(a, c, P):
    """
    Temperature of neutral, unheated gas at scale factor a: adiabatic cooling of a static
    uniform gas from T_init, T = T_init (a_i/a)^(3(gamma-1)), but not below temperature_floor.
    """
    gamma = float(P.get("gamma", 5.0 / 3.0))
    T_floor = float(P.get("temperature_floor", 0.0))
    return max(c["T_init"] * (c["a_i"] / a) ** (3.0 * (gamma - 1.0)), T_floor)


def temperature_check(T, xhii, a, c, P, start_tol=0.05, band=2.0):
    """
    Unit sanity check of the 'temperature' output.

    * First snapshot (a = a_i): the median temperature must equal T_init within start_tol.
      This tests the IC energy and the code->CGS energy conversion directly.
    * Later snapshots: the median temperature of neutral gas (x_HII < 0.1) is compared with
      adiabatic cooling from T_init and must agree within a factor `band`. Compton coupling
      and cooling cause modest departures; a unit error causes orders of magnitude.
    * Ionized gas (x_HII > 0.9) must lie in T_IONIZED_RANGE.

    Returns a dict of results; status values are "PASS", "FAIL", "WARN" or "n/a".
    """
    T = np.asarray(T, dtype=np.float64)
    neutral, ionized = xhii < NEUTRAL_XHII, xhii > IONIZED_XHII
    r = dict(
        a=a,
        T_init=c["T_init"],
        at_start=abs(a / c["a_i"] - 1.0) < 1.0e-3,
        T_expected_neutral=expected_neutral_temperature(a, c, P),
        n_neutral=int(neutral.sum()),
        n_ionized=int(ionized.sum()),
        n_cells=T.size,
        T_neutral=float(np.median(T[neutral])) if neutral.any() else float("nan"),
        T_ionized=float(np.median(T[ionized])) if ionized.any() else float("nan"),
        T_ionized_mean=float(T[ionized].mean()) if ionized.any() else float("nan"),
        T_min=float(T.min()),
        T_max=float(T.max()),
    )
    if r["at_start"]:
        T_all = float(np.median(T))
        r["T_start"] = T_all
        r["neutral_status"] = (
            "PASS" if abs(T_all / c["T_init"] - 1.0) <= start_tol else "FAIL"
        )
    elif r["n_neutral"] > 0:
        ratio = r["T_neutral"] / r["T_expected_neutral"]
        r["neutral_status"] = "PASS" if 1.0 / band <= ratio <= band else "WARN"
    else:
        r["neutral_status"] = "n/a"
    if r["n_ionized"] > 0:
        lo, hi = T_IONIZED_RANGE
        r["ionized_status"] = "PASS" if lo <= r["T_ionized"] <= hi else "WARN"
    else:
        r["ionized_status"] = "n/a"
    return r


def print_temperature_check(r):
    print("=== Temperature check ===")
    print(f"  T range in file: {r['T_min']:.4g} - {r['T_max']:.4g} K")
    if r["at_start"]:
        print(
            f"  first snapshot: median T = {r['T_start']:.4g} K, T_init = {r['T_init']:.4g} K"
            f"  -> {r['neutral_status']}"
        )
    else:
        print(
            f"  neutral gas ({r['n_neutral']} cells, x_HII < {NEUTRAL_XHII}): median T = {r['T_neutral']:.4g} K;"
            f" adiabatic from T_init = {r['T_expected_neutral']:.4g} K  -> {r['neutral_status']}"
        )
    if r["n_ionized"] > 0:
        lo, hi = T_IONIZED_RANGE
        print(
            f"  ionized gas ({r['n_ionized']} cells, x_HII > {IONIZED_XHII}): median T = {r['T_ionized']:.4g} K,"
            f" mean {r['T_ionized_mean']:.4g} K; expected {lo:.0e}-{hi:.0e} K  -> {r['ionized_status']}"
        )
    else:
        print("  ionized gas: none (no cells with x_HII > %.1f)" % IONIZED_XHII)
    if "FAIL" in (r["neutral_status"], r["ionized_status"]) or "WARN" in (
        r["neutral_status"],
        r["ionized_status"],
    ):
        print(
            "  -> temperatures far from these expectations usually mean an energy-unit inconsistency"
            " between the ICs, the chemistry and the output (see Compute_Gas_Temperature)."
        )


def expected_radii_at_temperature(a, P, T, ndot=NDOT):
    """
    Analytic radii with alpha_B evaluated at temperature T instead of T_init. Approximate:
    alpha_B is held fixed at T over the whole history.
    """
    P2 = dict(P)
    P2["T_init"] = T
    return expected_radii(a, derive_cosmology(P2, ndot=ndot))
