
import os, sys, json, math, time, random, logging, warnings
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

import networkx as nx
from shapely.geometry import Point, LineString
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 0: REPRODUCIBILITY & DIRECTORY SETUP
# ============================================================================

# Change this default value in PyCharm, or set the GNN_PPO_SEED environment
# variable. Results are isolated by seed and repeated runs never overwrite one
# another.
DEFAULT_RANDOM_SEED = 1024
try:
    RANDOM_SEED = int(os.environ.get("GNN_PPO_SEED", DEFAULT_RANDOM_SEED))
except ValueError as exc:
    raise ValueError("GNN_PPO_SEED must be an integer.") from exc

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

RESULTS_ROOT = "results/cfd_gnn_ppo_trilevel_yaw_v10_reliability"

def allocate_run_directory(results_root: str, seed: int) -> str:
    """Atomically allocate seed_<seed>/run_XX without overwriting old runs."""
    seed_dir = os.path.join(results_root, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    requested_name = os.environ.get("GNN_PPO_RUN_NAME", "").strip()
    if requested_name:
        candidate = os.path.join(seed_dir, requested_name)
        try:
            os.makedirs(candidate)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Requested run directory already exists: {candidate}"
            ) from exc
        return candidate

    run_index = 1
    while True:
        candidate = os.path.join(seed_dir, f"run_{run_index:02d}")
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            run_index += 1

RESULTS_DIR    = allocate_run_directory(RESULTS_ROOT, RANDOM_SEED)
PLOT_DATA_DIR  = os.path.join(RESULTS_DIR, "plot_data")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
for _d in [RESULTS_DIR, PLOT_DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(_d, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "run_metadata.json"), "w", encoding="utf-8") as _f:
    json.dump({
        "script": os.path.basename(__file__),
        "random_seed": RANDOM_SEED,
        "results_root": RESULTS_ROOT,
        "results_dir": RESULTS_DIR,
    }, _f, indent=2)


# ============================================================================
# SECTION 1: LOGGING INFRASTRUCTURE
# ============================================================================

def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger("WindFarmOpt")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

LOG = setup_logging(os.path.join(RESULTS_DIR, "run.log"))

def save_json(obj: dict, filename: str) -> None:
    path = os.path.join(PLOT_DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    LOG.info(f"JSON saved | {path}")

def log_section(title: str) -> None:
    sep = "=" * 72
    LOG.info(sep); LOG.info(f"  {title}"); LOG.info(sep)


# ============================================================================
# SECTION 2: GLOBAL CONFIGURATIONS
# ============================================================================

class DeviceConfig:
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = RANDOM_SEED

class ConstraintConfig:
    OBSTACLE_SAFETY_MARGIN = 80.0
    OBSTACLE_HARD_MARGIN   = 50.0
    CROSSING_PENALTY       = -40.0
    BOUNDARY_PADDING       = 200.0
    MAX_CROSSING_FOR_BEST  = 0

class EnvironmentConfig:
    LAYOUT_REPAIR_ITERS = 90
    LAYOUT_REPAIR_ATTEMPTS = 40
    NUM_TURBINES                = 25
    NUM_SUBSTATIONS             = 1
    GRID_SIZE                   = (6500, 5000)
    MAX_STEPS                   = 60
    RATED_POWER                 = 8.0          # MW
    ROTOR_DIAMETER              = 164.0        # m  (Vestas V164-8.0 MW)
    HUB_HEIGHT                  = 150.0        # m
    MIN_TURBINE_DISTANCE_FACTOR = 4.0
    AIR_DENSITY                 = 1.225
    Cp                          = 0.48
    ETA_GEAR                    = 0.95
    # Wake parameters
    WAKE_DECAY_CONSTANT         = 0.04
    THRUST_COEFFICIENT          = 0.75         # C_T at rated (used for load calc)
    INDUCTION_FACTOR            = 1 - math.sqrt(1 - 0.75)  # axial induction ≈ 0.5
    TI_AMBIENT                  = 0.06         # 6 % freestream TI (offshore typical)
    ABL_ALPHA                   = 0.12
    Z_HUB                       = 150.0
    Z_REF                       = 150.0
    # Active wake steering / yaw-control layer [v7]
    YAW_MAX_DEG                 = 25.0         # bounded continuous control signal
    YAW_POWER_EXPONENT          = 1.88         # P_yaw = P_0 cos(gamma)^alpha
    YAW_CT_EXPONENT             = 2.0          # C_T,yaw = C_T cos(gamma)^2
    WAKE_STEERING_GAIN          = 0.30         # empirical BPA centreline deflection gain
    YAW_TI_RELIEF_GAIN          = 0.35         # down-weight added TI from yawed wakes
    YAW_CONTROL_REWARD_WEIGHT   = 3000.0       # reward scale for TI relief and yaw effort
    YAW_EFFORT_PENALTY          = 0.015        # soft penalty per mean |yaw| / yaw_max
    # Electrical
    COLLECTION_VOLTAGE          = 33.0
    CABLE_RESISTANCE_PER_METER  = 0.00015
    POWER_FACTOR                = 0.95
    MAX_CABLE_LENGTH            = 4500
    FORBIDDEN_ZONES = [
        {'center': (1500, 1000), 'radius': 200},
        {'center': (3000, 2000), 'radius': 150},
    ]

class LifecycleConfig:
    FATIGUE_ROUTE_COST_PER_WEIGHTED_KM = 1_200_000.0
    FATIGUE_CRITICAL_ROUTE_COST_PER_KM = 240_000.0
    CABLE_COST_PER_KM    = 500_000.0
    ENERGY_PRICE         = 0.50
    DISCOUNT_RATE        = 0.05
    LIFESPAN             = 25
    OPERATING_HOURS      = 4_000
    # Fatigue-economics layer [v8]
    FATIGUE_REFERENCE_TI          = 0.12
    FATIGUE_DAMAGE_EXPONENT       = 4.0
    FATIGUE_COST_PER_DAMAGE_INDEX = 4_000_000.0
    FATIGUE_CRITICAL_TI           = 0.15
    YAW_ACTUATION_COST_PER_DEG_YEAR = 2_000.0

class RobustConfig:
    ROBUST_ROUTE_COST_PER_WEIGHTED_KM = 1_350_000.0
    ROBUST_LOW_POWER_ROUTE_COST_PER_KM = 300_000.0
    ENABLED = True
    # Deterministic quadrature scenarios: wind-speed delta, wind-direction delta,
    # yaw-tracking error amplitude, probability weight.
    SCENARIOS = [
        {'du':  0.0, 'ddir':   0.0, 'dyaw':  0.0, 'weight': 0.40},
        {'du': -1.0, 'ddir': -10.0, 'dyaw': -3.0, 'weight': 0.15},
        {'du': -1.0, 'ddir':  10.0, 'dyaw':  3.0, 'weight': 0.15},
        {'du':  1.0, 'ddir': -10.0, 'dyaw':  3.0, 'weight': 0.15},
        {'du':  1.0, 'ddir':  10.0, 'dyaw': -3.0, 'weight': 0.15},
    ]
    CVAR_ALPHA = 0.20
    POWER_SHORTFALL_PRICE_MULT = 1.0
    TI_CVAR_COST_MULT = 0.35

class ReliabilityConfig:
    ENABLED = True
    CABLE_FAILURE_RATE_PER_KM_YEAR = 0.015
    REPAIR_TIME_HOURS = 720.0
    UNSERVED_ENERGY_PRICE_MULT = 2.5
    N1_OUTAGE_TURBINE_PENALTY = 0.16
    REDUNDANCY_REWARD_WEIGHT = 240.0
    RELIABLE_MAX_N1_OUTAGE_TURBINES = 0
    RELIABLE_MIN_REDUNDANCY_RATIO = 1.0
    RELIABLE_MAX_ENS_MWH = 0.0
    RELIABLE_COST_TOLERANCE_M = 5.0
    # Third reliability fix: keep a near-reliable archive so final post-processing
    # can strengthen promising looped topologies instead of falling back to a radial tree.
    NEAR_RELIABLE_MAX_N1_OUTAGE_TURBINES = 10
    NEAR_RELIABLE_MIN_REDUNDANCY_RATIO = 0.25
    NEAR_RELIABLE_MAX_ENS_MWH = 3500.0
    AUGMENT_MAX_ADDITIONS = 14
    AUGMENT_CANDIDATE_LIMIT = 0  # 0 means evaluate all feasible add-edge candidates.

class GNNConfig:
    NODE_FEATURE_DIM   = 18
    EDGE_FEATURE_DIM   = 3
    HIDDEN_DIM         = 128
    OUTPUT_DIM         = 64
    DROPOUT            = 0.10
    HEADS              = 4
    PRETRAIN_EPOCHS    = 300
    PRETRAIN_BATCH_SIZE= 64
    PRETRAIN_LR        = 1e-3
    NUM_QUANTILES      = 51

class PPOConfig:
    LR_INIT          = 2e-4
    LR_FINAL         = 5e-5
    GAMMA            = 0.995
    GAE_LAMBDA       = 0.95
    CLIP_EPSILON     = 0.20
    VALUE_LOSS_COEF  = 0.50
    ENTROPY_COEF     = 0.03
    MAX_GRAD_NORM    = 0.50
    POLYAK_TAU       = 0.005
    K_EPOCHS         = 10
    BATCH_SIZE       = 128
    UPDATE_TIMESTEPS = 512

class TrainingConfig:
    NUM_EPISODES  = 2500
    SAVE_INTERVAL = 100

class CurriculumConfig:
    STAGES = [
        {
            'name': 'pure_connectivity',
            'min_episodes': 200,
            'cost_weight_target': 0.0,
            'reward_mode': 'pure_connectivity',
            'max_disconnections': 15,
            'max_steps_offset': 0,
            'advancement_criteria': {
                'min_success_rate':        0.10,
                'max_isolated':            0.50,
                'max_cost_cv':             1.00,
                'min_consecutive_success': 3,
            },
            'target_entropy': 3.0,
        },
        {
            'name': 'stable_connectivity',
            'min_episodes': 300,
            'cost_weight_target': 0.20,
            'reward_mode': 'stable_connectivity',
            'max_disconnections': 2,
            'max_steps_offset': 5,
            'advancement_criteria': {
                'min_success_rate':        0.60,
                'max_isolated':            0.20,
                'max_cost_cv':             0.30,
                'min_consecutive_success': 20,
            },
            'target_entropy': 1.2,
        },
        {
            'name': 'cost_aware',
            'min_episodes': 400,
            'cost_weight_target': 0.50,
            'reward_mode': 'cost_aware',
            'max_disconnections': 0,
            'max_steps_offset': 10,
            'advancement_criteria': {
                'min_success_rate':        0.85,
                'max_isolated':            0.00,
                'max_cost_cv':             0.15,
                'min_consecutive_success': 50,
            },
            'target_entropy': 0.6,
        },
        {
            'name': 'cost_optimisation',
            'min_episodes': 600,
            'cost_weight_target': 0.80,
            'reward_mode': 'cost_optimisation',
            'max_disconnections': 0,
            'max_steps_offset': 15,
            'advancement_criteria': None,
            'target_entropy': 0.2,
        },
    ]

class DataGenerationConfig:
    NUM_SAMPLES       = 800
    LAYOUT_CANDIDATES = 80
    NSGA_GENERATIONS  = 60


# ============================================================================
# SECTION 3: UTILITIES – EARLY STOPPING & LR SCHEDULER
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6, restore_best=True):
        self.patience=patience; self.min_delta=min_delta; self.restore_best=restore_best
        self.best_loss=float('inf'); self.counter=0; self.best_weights=None; self.triggered=False

    def __call__(self, val_loss: float, model=None) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss=val_loss; self.counter=0
            if model is not None and self.restore_best:
                self.best_weights = {k: v.clone().detach() for k,v in model.named_parameters()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                if model is not None and self.restore_best and self.best_weights:
                    for name, param in model.named_parameters():
                        if name in self.best_weights:
                            param.data.copy_(self.best_weights[name])
        return self.triggered

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.5, patience=10, min_lr=1e-6):
        self.optimizer=optimizer; self.factor=factor; self.patience=patience
        self.min_lr=min_lr; self.best=float('inf'); self.counter=0

    def step(self, metric: float):
        if metric < self.best - 1e-4:
            self.best=metric; self.counter=0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for pg in self.optimizer.param_groups:
                    new_lr = max(pg['lr']*self.factor, self.min_lr)
                    if new_lr < pg['lr']:
                        LOG.info(f"LRScheduler | LR {pg['lr']:.2e} → {new_lr:.2e}")
                        pg['lr'] = new_lr
                self.counter = 0


# ============================================================================
# SECTION 4: CFD WAKE SIMULATORS  [Innovation I1]
# ============================================================================

def _frandsen_ti(TI_0: float, CT: float, S: float) -> float:
    """
    Frandsen (2005) added turbulence intensity at distance S (in rotor diameters).
    TI_add = 0.9 * A^0.7 / (S/D)^0.7  where A = axial induction factor.
    Combined: TI_total = sqrt(TI_0^2 + TI_add^2)
    """
    a = 1.0 - math.sqrt(max(0.0, 1.0 - CT))
    TI_add = 0.9 * (a ** 0.7) / max(S, 0.1) ** 0.7
    return math.sqrt(TI_0 ** 2 + TI_add ** 2)


class JensenWakeModel:
    """
    Jensen (1983) top-hat analytical wake model (baseline).
    """
    def __init__(self, domain_bounds=None):
        self.rho     = EnvironmentConfig.AIR_DENSITY
        self.u_inlet = 10.0
        self.D       = EnvironmentConfig.ROTOR_DIAMETER
        self.C_T     = EnvironmentConfig.THRUST_COEFFICIENT
        self.k       = EnvironmentConfig.WAKE_DECAY_CONSTANT
        self.Cp      = EnvironmentConfig.Cp
        self.eta     = EnvironmentConfig.ETA_GEAR
        self.rated   = EnvironmentConfig.RATED_POWER
        self.n_calls = 0
        self.total_t = 0.0

    def compute_power(self, x: np.ndarray, y: np.ndarray) -> float:
        t0  = time.perf_counter()
        n   = len(x)
        R   = self.D / 2.0
        a   = 1.0 - math.sqrt(max(0.0, 1.0 - self.C_T))
        u   = np.full(n, self.u_inlet)
        order = np.argsort(x)
        for i in range(n):
            curr = order[i]; dsq = 0.0
            for j in range(i):
                up = order[j]
                dx = x[curr] - x[up]
                if dx <= 0: continue
                r_wake = R + self.k * dx
                dy = abs(y[curr] - y[up])
                if dy > r_wake: continue
                fraction = min(1.0, (r_wake - dy) / (2*R))
                dsq += (2*a*(R/r_wake)**2 * fraction) ** 2
            u[curr] *= max(0.0, 1.0 - np.sqrt(min(dsq, 0.99)))
        A = math.pi * R**2
        total = sum(min(0.5*self.rho*A*u[i]**3*self.Cp*self.eta/1e6, self.rated)
                    for i in range(n))
        self.n_calls += 1; self.total_t += time.perf_counter()-t0
        return total

    def get_stats(self):
        avg = self.total_t / max(self.n_calls, 1)
        return {'n_calls': self.n_calls, 'total_time_s': round(self.total_t, 4),
                'avg_time_s': round(avg, 6)}


class EnhancedCFDModel:

    def __init__(self, domain_bounds=None):
        self.rho        = EnvironmentConfig.AIR_DENSITY
        self.u_inlet    = 10.0
        self.D          = EnvironmentConfig.ROTOR_DIAMETER
        self.C_T        = EnvironmentConfig.THRUST_COEFFICIENT
        self.k          = EnvironmentConfig.WAKE_DECAY_CONSTANT
        self.TI         = EnvironmentConfig.TI_AMBIENT
        self.ABL_alpha  = EnvironmentConfig.ABL_ALPHA
        self.Z_hub      = EnvironmentConfig.Z_HUB
        self.Z_ref      = EnvironmentConfig.Z_REF
        self.Cp         = EnvironmentConfig.Cp
        self.eta        = EnvironmentConfig.ETA_GEAR
        self.rated      = EnvironmentConfig.RATED_POWER
        self.n_calls    = 0
        self.total_t    = 0.0

    def _prepare_yaw_angles(self, yaw_angles, n: int) -> np.ndarray:
        if yaw_angles is None:
            return np.zeros(n, dtype=float)
        yaw = np.asarray(yaw_angles, dtype=float).reshape(-1)
        if yaw.size != n:
            raise ValueError(f"yaw_angles must contain {n} values, got {yaw.size}")
        return np.clip(yaw, -EnvironmentConfig.YAW_MAX_DEG, EnvironmentConfig.YAW_MAX_DEG)

    def _yaw_ct(self, gamma_rad: float) -> float:
        c = max(0.0, math.cos(gamma_rad))
        return float(np.clip(self.C_T * (c ** EnvironmentConfig.YAW_CT_EXPONENT), 1e-4, 0.999))

    def _yaw_power_factor(self, yaw_deg: np.ndarray) -> np.ndarray:
        gamma = np.radians(np.asarray(yaw_deg, dtype=float))
        return np.clip(np.cos(gamma), 0.0, 1.0) ** EnvironmentConfig.YAW_POWER_EXPONENT

    def _wake_center_offset(self, dx: float, gamma_rad: float, ct_yaw: float) -> float:
        """
        Lightweight BPA-compatible wake steering term.
        Positive yaw shifts the downstream Gaussian centreline laterally; the
        offset is damped with downstream distance to avoid unphysical far-wake drift.
        """
        if abs(gamma_rad) < 1e-8 or dx <= 0:
            return 0.0
        D = self.D
        attenuation = 1.0 / (1.0 + dx / (15.0 * D))
        ct_ratio = ct_yaw / max(self.C_T, 1e-6)
        return (EnvironmentConfig.WAKE_STEERING_GAIN * dx * math.sin(gamma_rad)
                * math.cos(gamma_rad) * ct_ratio * attenuation)

    def compute_power(self, x: np.ndarray, y: np.ndarray,
                      wind_speed: float = 10.0, wind_dir_deg: float = 0.0,
                      yaw_angles=None) -> float:
        t0    = time.perf_counter()
        n     = len(x)
        R     = self.D / 2.0
        TI    = self.TI
        k_eff = self.k * (1.0 + 0.5*(TI + 0.1*math.sqrt(0.8)))  # TI-amplified decay
        yaw_deg = self._prepare_yaw_angles(yaw_angles, n)
        yaw_rad = np.radians(yaw_deg)
        yaw_pf  = self._yaw_power_factor(yaw_deg)

        theta = math.radians(wind_dir_deg)
        c, s  = math.cos(theta), math.sin(theta)
        rot   = np.array([[c, -s], [s, c]])
        xy    = np.column_stack([x, y])
        rot_  = xy @ rot.T
        order = np.argsort(rot_[:, 0])

        hc    = (self.Z_hub/self.Z_ref) ** self.ABL_alpha   # ABL height correction
        u_eff = np.full(n, wind_speed)

        for i in range(n):
            curr = order[i]; dsq = 0.0
            for j in range(i):
                up = order[j]
                dx = rot_[curr, 0] - rot_[up, 0]
                dy = rot_[curr, 1] - rot_[up, 1]
                if dx <= 0: continue
                ct_yaw = self._yaw_ct(float(yaw_rad[up]))
                dy -= self._wake_center_offset(dx, float(yaw_rad[up]), ct_yaw)
                sigma   = k_eff * dx + R
                # Wake meandering sinusoidal correction (empirical, Larsen 2008)
                meander = 1.0 + 0.10 * math.sin(2.0*math.pi*dx/(10.0*R))
                gauss   = math.exp(-(dy**2) / (2.0*sigma**2))
                deficit = ((1.0 - math.sqrt(max(0.0, 1.0 - ct_yaw)))
                           * (R/(R + k_eff*dx))**2 * gauss * meander)
                dsq    += deficit ** 2
            vel_ratio     = 1.0 - math.sqrt(min(dsq, 0.98))
            u_eff[curr]   = wind_speed * max(0.0, vel_ratio) * hc
            # (power summed outside loop to keep u_eff array intact)

        total = sum(
            min(0.5*self.rho*math.pi*R**2*u_eff[i]**3*self.Cp*self.eta*yaw_pf[i]/1e6, self.rated)
            for i in range(n))
        self.n_calls += 1; self.total_t += time.perf_counter()-t0
        return total

    def compute_power_with_ti(self, x, y, wind_speed=10.0, wind_dir_deg=0.0,
                              yaw_angles=None):

        n = len(x); R = self.D/2.0
        TI_0 = self.TI
        k_eff = self.k*(1.0 + 0.5*(TI_0 + 0.1*math.sqrt(0.8)))
        yaw_deg = self._prepare_yaw_angles(yaw_angles, n)
        yaw_rad = np.radians(yaw_deg)
        yaw_pf  = self._yaw_power_factor(yaw_deg)
        theta = math.radians(wind_dir_deg)
        c, s  = math.cos(theta), math.sin(theta)
        rot   = np.array([[c,-s],[s,c]])
        rot_  = np.column_stack([x,y]) @ rot.T
        order = np.argsort(rot_[:,0])
        hc    = (self.Z_hub/self.Z_ref)**self.ABL_alpha
        u_eff = np.full(n, wind_speed)
        TI_eff= np.full(n, TI_0)

        for i in range(n):
            curr = order[i]; dsq = 0.0
            # [v5-1] Start with ambient TI variance; sum added variances in quadrature
            ti_add_sq = 0.0
            for j in range(i):
                up = order[j]
                dx = rot_[curr,0]-rot_[up,0]; dy = rot_[curr,1]-rot_[up,1]
                if dx <= 0: continue
                ct_yaw = self._yaw_ct(float(yaw_rad[up]))
                dy -= self._wake_center_offset(dx, float(yaw_rad[up]), ct_yaw)
                sigma   = k_eff*dx + R
                meander = 1.0 + 0.10*math.sin(2.0*math.pi*dx/(10.0*R))
                gauss   = math.exp(-(dy**2)/(2.0*sigma**2))
                deficit = ((1.0-math.sqrt(max(0.0,1.0-ct_yaw)))
                           *(R/(R+k_eff*dx))**2 * gauss * meander)
                dsq    += deficit**2
                # [v5-1] Frandsen added TI variance from turbine j (quadrature sum)
                S_D = dx / (R*2)  # distance in rotor diameters
                ti_total_j = _frandsen_ti(TI_0, ct_yaw, max(S_D, 0.5))
                overlap = float(np.clip(gauss, 0.0, 1.0))
                yaw_relief = 1.0 - EnvironmentConfig.YAW_TI_RELIEF_GAIN * abs(math.sin(float(yaw_rad[up])))
                ti_add_sq += max(0.0, ti_total_j**2 - TI_0**2) * overlap * max(0.0, yaw_relief)
            u_eff[curr]  = wind_speed*max(0.0,1.0-math.sqrt(min(dsq,0.98)))*hc
            TI_eff[curr] = math.sqrt(TI_0**2 + ti_add_sq)  # quadrature total

        total = sum(min(0.5*self.rho*math.pi*R**2*u_eff[i]**3*self.Cp*self.eta/1e6,
                        self.rated) * yaw_pf[i] for i in range(n))
        return total, u_eff, TI_eff

    def get_stats(self):
        avg = self.total_t/max(self.n_calls,1)
        return {'n_calls': self.n_calls, 'total_time_s': round(self.total_t,4),
                'avg_time_s': round(avg,6)}


# ============================================================================
# SECTION 5: META-HEURISTIC OPTIMISERS  [Innovation I6 – Baselines]
# ============================================================================

class _BaseOptimizer:
    def __init__(self, sim, n_turbines, domain_bounds, use_cfd=False):
        self.sim=sim; self.n_turbines=n_turbines; self.bounds=domain_bounds; self.use_cfd=use_cfd
        self.min_spacing = 3 * sim.D   # [v3-1] now 3×164m = 492m

    def _fitness(self, x, y):
        penalty = 0.0
        for i in range(self.n_turbines):
            for j in range(i+1, self.n_turbines):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if d < self.min_spacing:
                    penalty += (self.min_spacing-d)**2
        return self.sim.compute_power(x, y) - 1e5*penalty

    def _random_positions(self):
        m = 100
        return (np.random.uniform(self.bounds['x'][0]+m, self.bounds['x'][1]-m, self.n_turbines),
                np.random.uniform(self.bounds['y'][0]+m, self.bounds['y'][1]-m, self.n_turbines))

    def _clip(self, x, y):
        m = 100
        return (np.clip(x, self.bounds['x'][0]+m, self.bounds['x'][1]-m),
                np.clip(y, self.bounds['y'][0]+m, self.bounds['y'][1]-m))


class WhaleOptimizer(_BaseOptimizer):
    def __init__(self, sim, n_turbines, domain_bounds, n_whales=30, use_cfd=False):
        super().__init__(sim, n_turbines, domain_bounds, use_cfd)
        self.n_whales = n_whales

    def optimize(self, iterations=100):
        pop   = [self._random_positions() for _ in range(self.n_whales)]
        fits  = [self._fitness(x,y) for x,y in pop]
        bi    = int(np.argmax(fits))
        bp, bf = pop[bi], fits[bi]
        history=[]; tag="WOA-CFD" if self.use_cfd else "WOA-Jensen"
        for it in range(iterations):
            a=2-it*(2/iterations)
            for i in range(self.n_whales):
                r=np.random.rand(); A=2*a*r-a; C=2*r; l=np.random.uniform(-1,1)
                x,y=pop[i]; bx,by=bp
                if np.random.rand()<0.5:
                    if abs(A)<1:
                        x=bx-A*abs(C*bx-x); y=by-A*abs(C*by-y)
                    else:
                        ri=np.random.randint(self.n_whales); rx,ry=pop[ri]
                        x=rx-A*abs(C*rx-x); y=ry-A*abs(C*ry-y)
                else:
                    x=abs(bx-x)*np.exp(l)*np.cos(2*np.pi*l)+bx
                    y=abs(by-y)*np.exp(l)*np.cos(2*np.pi*l)+by
                x,y=self._clip(x,y); pop[i]=(x,y); f=self._fitness(x,y)
                if f>bf: bf=f; bp=(x,y)
            history.append(float(self.sim.compute_power(bp[0],bp[1])))
            if (it+1)%25==0: LOG.info(f"  {tag} iter={it+1:3d} | power={history[-1]:.3f} MW")
        return bp, history


class GreyWolfOptimizer(_BaseOptimizer):
    def __init__(self, sim, n_turbines, domain_bounds, n_wolves=30, use_cfd=False):
        super().__init__(sim, n_turbines, domain_bounds, use_cfd)
        self.n_wolves = n_wolves

    def optimize(self, iterations=100):
        pop   = [self._random_positions() for _ in range(self.n_wolves)]
        fits  = [self._fitness(x,y) for x,y in pop]
        idx   = np.argsort(fits)[::-1]
        alpha, beta, delta = pop[idx[0]], pop[idx[1]], pop[idx[2]]
        history=[]; tag="GWO-CFD" if self.use_cfd else "GWO-Jensen"
        for it in range(iterations):
            a=2-it*(2/iterations)
            new_pop=[]
            for x,y in pop:
                pos=[]
                for leader in [alpha,beta,delta]:
                    r1,r2=np.random.rand(2); A_l=2*a*r1-a; C_l=2*r2; lx,ly=leader
                    pos.append((lx-A_l*abs(C_l*lx-x), ly-A_l*abs(C_l*ly-y)))
                nx_=sum(p[0] for p in pos)/3; ny_=sum(p[1] for p in pos)/3
                nx_,ny_=self._clip(nx_,ny_); new_pop.append((nx_,ny_))
            pop=new_pop; fits=[self._fitness(x,y) for x,y in pop]
            idx=np.argsort(fits)[::-1]; alpha,beta,delta=pop[idx[0]],pop[idx[1]],pop[idx[2]]
            history.append(float(self.sim.compute_power(alpha[0],alpha[1])))
            if (it+1)%25==0: LOG.info(f"  {tag} iter={it+1:3d} | power={history[-1]:.3f} MW")
        return alpha, history


class GeneticAlgorithm(_BaseOptimizer):
    def __init__(self, sim, n_turbines, domain_bounds, pop_size=30, use_cfd=False):
        super().__init__(sim, n_turbines, domain_bounds, use_cfd)
        self.pop_size = pop_size

    def optimize(self, iterations=100):
        pop  = [self._random_positions() for _ in range(self.pop_size)]
        fits = [self._fitness(x,y) for x,y in pop]
        history=[]; tag="GA-CFD" if self.use_cfd else "GA-Jensen"
        for it in range(iterations):
            new_pop=[]
            while len(new_pop) < self.pop_size:
                p1=pop[np.random.choice(np.argsort(fits)[::-1][:10])]
                p2=pop[np.random.choice(np.argsort(fits)[::-1][:10])]
                alp=np.random.rand()
                cx=alp*p1[0]+(1-alp)*p2[0]; cy=alp*p1[1]+(1-alp)*p2[1]
                if np.random.rand()<0.3:
                    cx+=np.random.randn(self.n_turbines)*50
                    cy+=np.random.randn(self.n_turbines)*50
                cx,cy=self._clip(cx,cy); new_pop.append((cx,cy))
            pop=new_pop; fits=[self._fitness(x,y) for x,y in pop]
            best=pop[np.argmax(fits)]
            history.append(float(self.sim.compute_power(best[0],best[1])))
            if (it+1)%25==0: LOG.info(f"  {tag} iter={it+1:3d} | power={history[-1]:.3f} MW")
        return pop[np.argmax(fits)], history


class ParticleSwarmOptimizer(_BaseOptimizer):
    def __init__(self, sim, n_turbines, domain_bounds, n_particles=30, use_cfd=False):
        super().__init__(sim, n_turbines, domain_bounds, use_cfd)
        self.n_particles = n_particles

    def optimize(self, iterations=100):
        m=100
        xmin,xmax=self.bounds['x'][0]+m,self.bounds['x'][1]-m
        ymin,ymax=self.bounds['y'][0]+m,self.bounds['y'][1]-m
        pos=[(np.random.uniform(xmin,xmax,self.n_turbines),
              np.random.uniform(ymin,ymax,self.n_turbines)) for _ in range(self.n_particles)]
        vel=[(np.zeros(self.n_turbines),np.zeros(self.n_turbines)) for _ in range(self.n_particles)]
        pbest=list(pos); pbest_f=[self._fitness(x,y) for x,y in pos]
        gbest=pbest[int(np.argmax(pbest_f))]; history=[]; tag="PSO-CFD" if self.use_cfd else "PSO-Jensen"
        for it in range(iterations):
            w=0.9-it*(0.5/iterations); c1=c2=2.0
            new_pos=[]; new_vel=[]
            for i in range(self.n_particles):
                x,y=pos[i]; vx,vy=vel[i]; px,py=pbest[i]; gx,gy=gbest
                r1,r2=np.random.rand(2)
                nvx=w*vx+c1*r1*(px-x)+c2*r2*(gx-x)
                nvy=w*vy+c1*r1*(py-y)+c2*r2*(gy-y)
                nx_=np.clip(x+nvx,xmin,xmax); ny_=np.clip(y+nvy,ymin,ymax)
                new_pos.append((nx_,ny_)); new_vel.append((nvx,nvy))
                f=self._fitness(nx_,ny_)
                if f>pbest_f[i]:
                    pbest_f[i]=f; pbest[i]=(nx_,ny_)
                    if f>self._fitness(gbest[0],gbest[1]): gbest=(nx_,ny_)
            pos=new_pos; vel=new_vel
            history.append(float(self.sim.compute_power(gbest[0],gbest[1])))
            if (it+1)%25==0: LOG.info(f"  {tag} iter={it+1:3d} | power={history[-1]:.3f} MW")
        return gbest, history


class SimulatedAnnealing(_BaseOptimizer):
    def optimize(self, iterations=100):
        x,y=self._random_positions(); cur_f=self._fitness(x,y)
        best_x,best_y=x.copy(),y.copy(); best_f=cur_f
        history=[]; tag="SA-CFD" if self.use_cfd else "SA-Jensen"; T0,Tf=1e5,1.0
        for it in range(iterations):
            T=T0*(Tf/T0)**(it/iterations)
            nx_=x+np.random.randn(self.n_turbines)*100; ny_=y+np.random.randn(self.n_turbines)*100
            nx_,ny_=self._clip(nx_,ny_); nf=self._fitness(nx_,ny_); dE=nf-cur_f
            if dE>0 or np.random.rand()<np.exp(dE/(T+1e-12)): x,y=nx_,ny_; cur_f=nf
            if cur_f>best_f: best_f=cur_f; best_x=x.copy(); best_y=y.copy()
            history.append(float(self.sim.compute_power(best_x,best_y)))
            if (it+1)%25==0: LOG.info(f"  {tag} iter={it+1:3d} | power={history[-1]:.3f} MW | T={T:.1f}")
        return (best_x,best_y), history


class BayesianOptimizer(_BaseOptimizer):
    def optimize(self, n_init=5, n_iter=25):
        m=100
        xmin,xmax=self.bounds['x'][0]+m,self.bounds['x'][1]-m
        ymin,ymax=self.bounds['y'][0]+m,self.bounds['y'][1]-m
        D=2*self.n_turbines
        lb=np.hstack([np.full(self.n_turbines,xmin), np.full(self.n_turbines,ymin)])
        ub=np.hstack([np.full(self.n_turbines,xmax), np.full(self.n_turbines,ymax)])
        X_obs,Y_obs=[],[]
        for _ in range(n_init):
            x,y=self._random_positions(); f=self._fitness(x,y)
            X_obs.append(np.hstack([x,y])); Y_obs.append(f)
        kernel=ConstantKernel(1.0)*Matern(nu=2.5)
        gp=GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        best_x=X_obs[int(np.argmax(Y_obs))]; best_f=max(Y_obs); history=[]
        tag="BO-CFD" if self.use_cfd else "BO-Jensen"
        for it in range(n_iter):
            gp.fit(np.array(X_obs), np.array(Y_obs))
            def neg_ei(z):
                z=z.reshape(1,-1); mu,sig=gp.predict(z, return_std=True); sig=max(sig[0],1e-9)
                Z=(mu[0]-best_f)/sig; ei=sig*(Z*norm.cdf(Z)+norm.pdf(Z)); return -ei
            bz=None; bf=None
            for _ in range(10):
                z0=lb+np.random.rand(D)*(ub-lb)
                from scipy.optimize import minimize as sp_min
                res=sp_min(neg_ei, z0, bounds=list(zip(lb,ub)), method='L-BFGS-B')
                if bf is None or res.fun<bf: bf=res.fun; bz=res.x
            z=bz; x=z[:self.n_turbines]; y=z[self.n_turbines:]
            x,y=self._clip(x,y); f=self._fitness(x,y)
            X_obs.append(np.hstack([x,y])); Y_obs.append(f)
            if f>best_f: best_f=f; best_x=np.hstack([x,y])
            history.append(float(self.sim.compute_power(x,y)))
            if (it+1)%10==0: LOG.info(f"  {tag} iter={it+1:3d} | power={history[-1]:.3f} MW | n_obs={len(X_obs)}")
        bx=best_x[:self.n_turbines]; by=best_x[self.n_turbines:]
        return (bx,by), history


# ============================================================================
# SECTION 6: BASELINE COMPARISON RUNNER  [Innovation I6]
# ============================================================================

def run_baseline_comparison(n_turbines, domain_bounds):
    log_section("Phase 0 – Comprehensive Baseline Comparison  [I6]")
    jensen_sim = JensenWakeModel(domain_bounds)
    cfd_sim    = EnhancedCFDModel(domain_bounds)

    algos_j = {
        'WOA': WhaleOptimizer(jensen_sim, n_turbines, domain_bounds, n_whales=30),
        'GWO': GreyWolfOptimizer(jensen_sim, n_turbines, domain_bounds, n_wolves=30),
        'GA' : GeneticAlgorithm(jensen_sim, n_turbines, domain_bounds, pop_size=30),
        'PSO': ParticleSwarmOptimizer(jensen_sim, n_turbines, domain_bounds, n_particles=30),
        'SA' : SimulatedAnnealing(jensen_sim, n_turbines, domain_bounds),
        'BO' : BayesianOptimizer(jensen_sim, n_turbines, domain_bounds),
    }
    algos_c = {
        'WOA': WhaleOptimizer(cfd_sim, n_turbines, domain_bounds, n_whales=30, use_cfd=True),
        'GWO': GreyWolfOptimizer(cfd_sim, n_turbines, domain_bounds, n_wolves=30, use_cfd=True),
        'GA' : GeneticAlgorithm(cfd_sim, n_turbines, domain_bounds, pop_size=30, use_cfd=True),
        'PSO': ParticleSwarmOptimizer(cfd_sim, n_turbines, domain_bounds, n_particles=30, use_cfd=True),
        'SA' : SimulatedAnnealing(cfd_sim, n_turbines, domain_bounds, use_cfd=True),
        'BO' : BayesianOptimizer(cfd_sim, n_turbines, domain_bounds, use_cfd=True),
    }

    results_j, results_c = {}, {}
    LOG.info("--- Part A: Jensen Model (Analytical Baseline) ---")
    best_layout_j=None; best_power_j=-1.0
    for name, opt in algos_j.items():
        LOG.info(f"[Jensen] Running {name} ...")
        t0=time.perf_counter()
        (bx,by),hist = opt.optimize(n_init=5,n_iter=25) if name=='BO' else opt.optimize(iterations=100)
        t1=time.perf_counter(); p_mw=jensen_sim.compute_power(bx,by)
        LOG.info(f"  [Jensen] {name} done | power={p_mw:.3f} MW | evals={len(hist)} | time={t1-t0:.2f}s")
        results_j[name]={'power_mw': round(p_mw,4), 'n_evals': len(hist),
                         'time_s': round(t1-t0,2), 'history': hist}
        if p_mw>best_power_j: best_power_j=p_mw; best_layout_j=np.column_stack([bx,by])

    LOG.info("--- Part B: Enhanced CFD Model (High-Fidelity) ---")
    best_layout_c=None; best_power_c=-1.0
    for name, opt in algos_c.items():
        LOG.info(f"[CFD] Running {name} ...")
        t0=time.perf_counter()
        (bx,by),hist = opt.optimize(n_init=5,n_iter=25) if name=='BO' else opt.optimize(iterations=100)
        t1=time.perf_counter(); p_mw=cfd_sim.compute_power(bx,by)
        LOG.info(f"  [CFD] {name} done | power={p_mw:.3f} MW | evals={len(hist)} | time={t1-t0:.2f}s")
        results_c[name]={'power_mw': round(p_mw,4), 'n_evals': len(hist),
                         'time_s': round(t1-t0,2), 'history': hist}
        if p_mw>best_power_c: best_power_c=p_mw; best_layout_c=np.column_stack([bx,by])

    results_c['_cfd_stats']=cfd_sim.get_stats()
    imps=[((results_c[k]['power_mw']-results_j[k]['power_mw'])
           /max(results_j[k]['power_mw'],1e-8)*100)
          for k in results_j if k in results_c]
    avg_imp=float(np.mean(imps)) if imps else None
    if avg_imp is not None:
        LOG.info(f"  CFD vs Jensen avg AEP improvement: {avg_imp:+.2f}%")
    return results_j, results_c, best_layout_c, avg_imp


# ============================================================================
# SECTION 7: OFFSHORE WIND ENVIRONMENT
# ============================================================================

class OffshoreWindEnv:
    """
    Graph-based MDP for cable topology optimisation.

    State  : (node_features [N×18], combined_edge_index [2×E], edge_attr [E×3])
    Action : ((u, v, act_type), yaw_angles[N]) with act_type in {0=remove, 1=add}
    Reward : shaped per reward_mode (pure_connectivity → cost_optimisation)

    """
    def __init__(self, initial_layout=None):
        self.num_turbines = EnvironmentConfig.NUM_TURBINES
        self.num_nodes    = self.num_turbines + EnvironmentConfig.NUM_SUBSTATIONS
        self.max_steps    = EnvironmentConfig.MAX_STEPS
        self.wind_speed   = 10.0
        self.wind_direction = 0.0
        self.cost_weight  = 0.0
        self.reward_mode  = 'pure_connectivity'
        self.max_allowed_disconnections = 15
        self.best_cost_window = deque(maxlen=100)  # [v3-8] rolling best cost
        # [v4-2] Dynamic cost reward normalisation: EMA of observed costs
        self._cost_ema: float = None        # type: ignore[assignment]
        self._cost_std_ema: float = 15.0    # initial uncertainty (M RMB)

        self._init_positions(initial_layout)
        self._cfd_model = EnhancedCFDModel()
        self.yaw_angles = np.zeros(self.num_turbines, dtype=float)
        self.wake_cache = {'is_valid': False,
                          'effective_velocities': np.zeros(self.num_turbines),
                          'ti_effective': np.full(self.num_turbines, EnvironmentConfig.TI_AMBIENT),
                          'yaw_power_factor': np.ones(self.num_turbines),
                          'turbine_power': np.zeros(self.num_turbines),
                          'total_power': 0.0}
        self.node_feats = torch.zeros(self.num_nodes, GNNConfig.NODE_FEATURE_DIM)
        self.reset()

    def _init_positions(self, layout=None):
        W, H  = EnvironmentConfig.GRID_SIZE
        pad   = ConstraintConfig.BOUNDARY_PADDING
        rng   = np.random.RandomState(RANDOM_SEED)
        if layout is not None:
            positions = layout[:self.num_turbines].copy()
        else:
            positions = rng.rand(self.num_turbines, 2)*np.array([W-2*pad, H-2*pad]) + pad
        sub_pos = np.array([[W/2.0, H/2.0]])
        self.node_positions       = np.vstack([positions, sub_pos])
        self.turbine_positions    = self.node_positions[:self.num_turbines]
        self.substation_positions = self.node_positions[self.num_turbines:]
        self._clamp_turbines()

    def _clamp_turbines(self):
        W, H = EnvironmentConfig.GRID_SIZE; pad = ConstraintConfig.BOUNDARY_PADDING
        self.node_positions[:self.num_turbines, 0] = np.clip(
            self.node_positions[:self.num_turbines,0], pad, W-pad)
        self.node_positions[:self.num_turbines, 1] = np.clip(
            self.node_positions[:self.num_turbines,1], pad, H-pad)
        self.turbine_positions = self.node_positions[:self.num_turbines]

    def reset(self, use_mst_init=True):
        self.current_step     = 0
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.edge_index       = torch.empty((2,0), dtype=torch.long)
        self.edge_attr        = torch.empty((0, GNNConfig.EDGE_FEATURE_DIM-1), dtype=torch.float32)
        self.yaw_angles       = np.zeros(self.num_turbines, dtype=float)
        self._clamp_turbines()
        self.wake_cache['is_valid'] = False
        self._precompute_wake()
        if use_mst_init:
            self._build_mst_topology()
        self._update_node_features()
        self.prev_isolated_count = self._count_isolated_turbines()
        self.virtual_edge_index  = self._build_knn_graph()
        return self._get_state()

    def _build_mst_topology(self):
        G = nx.Graph()
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                d = float(np.linalg.norm(self.node_positions[i]-self.node_positions[j]))
                if d <= EnvironmentConfig.MAX_CABLE_LENGTH:
                    G.add_edge(i, j, weight=d)
        if G.number_of_nodes() < self.num_nodes: return
        try:
            mst = nx.minimum_spanning_tree(G)
            for u, v in mst.edges(): self._add_edge(u, v)
        except Exception: pass

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def _build_knn_graph(self, k=5):
        edges = []
        for i in range(self.num_nodes):
            dists = sorted([(np.linalg.norm(self.node_positions[i]-self.node_positions[j]),j)
                            for j in range(self.num_nodes) if j!=i])
            for _, j in dists[:k]: edges.append([i,j])
        return torch.tensor(edges, dtype=torch.long).t()

    def _get_state(self):
        n_virtual = self.virtual_edge_index.shape[1]
        n_actual  = self.edge_index.shape[1]
        if n_actual > 0:
            actual_flag = torch.zeros(n_actual, 1, dtype=torch.float32)
            actual_ea   = torch.cat([self.edge_attr, actual_flag], dim=1)
        else:
            actual_ea = torch.empty((0, GNNConfig.EDGE_FEATURE_DIM), dtype=torch.float32)
        virtual_ea = torch.zeros(n_virtual, GNNConfig.EDGE_FEATURE_DIM, dtype=torch.float32)
        virtual_ea[:, 2] = 1.0
        combined_ei = torch.cat([self.virtual_edge_index, self.edge_index], dim=1)
        combined_ea = torch.cat([virtual_ea, actual_ea], dim=0)
        return (self.node_feats.clone(), combined_ei, combined_ea)

    def _get_unique_edges(self):
        edges=set(); rows,cols=np.where(self.adjacency_matrix>0)
        for r,c in zip(rows,cols):
            if r<c: edges.add((r,c))
        return list(edges)

    def _build_nx_graph(self):
        G = nx.Graph(); G.add_nodes_from(range(self.num_nodes))
        for u,v in self._get_unique_edges(): G.add_edge(u,v)
        return G

    def _get_connected_nodes(self):
        start=self.num_nodes-1; visited=set([start]); queue=deque([start])
        adj=defaultdict(list); rows,cols=np.where(self.adjacency_matrix>0)
        for r,c in zip(rows,cols): adj[r].append(c)
        while queue:
            u=queue.popleft()
            for v in adj[u]:
                if v not in visited: visited.add(v); queue.append(v)
        return visited

    def _count_isolated_turbines(self):
        connected=self._get_connected_nodes()
        return self.num_turbines - sum(1 for i in range(self.num_turbines) if i in connected)


    def _precompute_wake(self):
        _, u_eff, TI_eff = self._cfd_model.compute_power_with_ti(
            self.node_positions[:self.num_turbines,0],
            self.node_positions[:self.num_turbines,1],
            wind_speed=self.wind_speed,
            wind_dir_deg=self.wind_direction,
            yaw_angles=self.yaw_angles,
        )
        R = EnvironmentConfig.ROTOR_DIAMETER/2.0
        yaw_pf = self._cfd_model._yaw_power_factor(self.yaw_angles)
        turbine_power = np.array([
            min(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2
                *u_eff[i]**3*EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR*yaw_pf[i]/1e6,
                EnvironmentConfig.RATED_POWER)
            for i in range(self.num_turbines)
        ], dtype=float)
        total = float(np.sum(turbine_power))
        self.wake_cache = {'effective_velocities': u_eff, 'ti_effective': TI_eff,
                           'yaw_power_factor': yaw_pf,
                           'turbine_power': turbine_power,
                           'total_power': total, 'is_valid': True}


    def _apply_yaw_action(self, yaw_angles):
        if yaw_angles is None:
            return
        yaw = np.asarray(yaw_angles, dtype=float).reshape(-1)
        if yaw.size != self.num_turbines:
            raise ValueError(f"yaw action must contain {self.num_turbines} values, got {yaw.size}")
        self.yaw_angles = np.clip(yaw, -EnvironmentConfig.YAW_MAX_DEG, EnvironmentConfig.YAW_MAX_DEG)
        self.wake_cache['is_valid'] = False


    def _update_node_features(self):
        if not self.wake_cache['is_valid']: self._precompute_wake()
        W, H  = EnvironmentConfig.GRID_SIZE
        feats = np.zeros((self.num_nodes, GNNConfig.NODE_FEATURE_DIM))

        # Spatial features
        feats[:,0] = self.node_positions[:,0] / W
        feats[:,1] = self.node_positions[:,1] / H
        # Node type
        feats[:self.num_turbines, 2] = 1.0   # turbine flag
        feats[self.num_turbines:, 3] = 1.0   # substation flag
        # Connectivity status
        connected = self._get_connected_nodes()
        for idx in connected: feats[idx, 4] = 1.0
        # Wake-effective velocity
        eff_v = self.wake_cache['effective_velocities']
        feats[:self.num_turbines, 5] = eff_v / 25.0
        # Normalised power output
        feats[:self.num_turbines, 6] = np.clip((eff_v/12.0)**3, 0, 1)
        # Distance to substation (normalised)
        sub_pos = self.substation_positions[0]
        dists   = np.linalg.norm(self.node_positions-sub_pos, axis=1)
        grid_diag = np.linalg.norm(EnvironmentConfig.GRID_SIZE)
        feats[:,7] = dists / grid_diag
        # Mean cable length (normalised)
        for i in range(self.num_nodes):
            nbrs = np.where(self.adjacency_matrix[i]>0)[0]
            if len(nbrs)>0:
                lens = [np.linalg.norm(self.node_positions[i]-self.node_positions[j]) for j in nbrs]
                feats[i,8] = np.mean(lens) / 1000.0
        # Dijkstra electrical path length to substation
        feats[:,9] = self._dijkstra_feature()
        # Node degree (normalised)
        feats[:,10] = np.sum(self.adjacency_matrix, axis=1) / self.num_nodes
        # Downstream turbine count (wake exposure)
        feats[:,11] = self._downstream_feature()
        # [v3-3] Frandsen effective turbulence intensity at hub
        TI_eff = self.wake_cache['ti_effective']
        feats[:self.num_turbines, 12] = TI_eff / 0.25   # normalise by 25% TI ceiling
        # [v3-3] Wind-stream upstream score (proxy for shadowing)
        feats[:,13] = self._upstream_score_feature()
        # ABL shear-corrected velocity ratio
        hc = (EnvironmentConfig.Z_HUB/EnvironmentConfig.Z_REF)**EnvironmentConfig.ABL_ALPHA
        feats[:self.num_turbines, 14] = np.clip(eff_v*hc/25.0, 0, 1)
        # Cable crossing count (normalised) – reward-relevant feature
        feats[:,15] = self._crossing_feature()
        yaw_pf = self.wake_cache.get('yaw_power_factor', np.ones(self.num_turbines))
        feats[:self.num_turbines, 16] = self.yaw_angles / EnvironmentConfig.YAW_MAX_DEG
        feats[:self.num_turbines, 17] = yaw_pf

        self.node_feats = torch.tensor(feats, dtype=torch.float32)

    def _dijkstra_feature(self):
        G=nx.Graph(); sub_idx=self.num_nodes-1
        rows,cols=np.where(self.adjacency_matrix>0)
        for r,c in zip(rows,cols):
            if r<c:
                G.add_edge(r,c, weight=np.linalg.norm(self.node_positions[r]-self.node_positions[c]))
        grid_diag=np.linalg.norm(EnvironmentConfig.GRID_SIZE)
        default=np.linalg.norm(self.node_positions-self.node_positions[sub_idx],axis=1)/grid_diag
        try:
            if not G.has_node(sub_idx): return default
            lengths=nx.single_source_dijkstra_path_length(G, sub_idx, weight='weight')
            out=np.zeros(self.num_nodes)
            for i in range(self.num_nodes): out[i]=lengths.get(i, default[i]*grid_diag)
            return out/grid_diag
        except Exception: return default

    def _downstream_feature(self):
        counts=np.zeros(self.num_nodes)
        wd_rad=np.radians(self.wind_direction)
        wind_v=np.array([math.cos(wd_rad), math.sin(wd_rad)])
        for i in range(self.num_turbines):
            pi=self.node_positions[i]
            for j in range(self.num_turbines):
                if i==j: continue
                vij=self.node_positions[j]-pi; dist=np.linalg.norm(vij)
                if dist>2000: continue
                dot=np.dot(wind_v, vij)/(dist+1e-8)
                angle=np.arccos(np.clip(dot,-1.0,1.0))
                if angle<np.radians(20): counts[i]+=1
        return counts/5.0

    def _upstream_score_feature(self):
        """Score how upstream each turbine is (fewer waking turbines → higher score)."""
        scores=np.zeros(self.num_nodes)
        wd_rad=np.radians(self.wind_direction)
        wind_v=np.array([math.cos(wd_rad), math.sin(wd_rad)])
        for i in range(self.num_turbines):
            pi=self.node_positions[i]; shadow=0
            for j in range(self.num_turbines):
                if i==j: continue
                vij=pi-self.node_positions[j]; dist=np.linalg.norm(vij)
                if dist>3000: continue
                dot=np.dot(wind_v,vij)/(dist+1e-8)
                angle=np.arccos(np.clip(dot,-1,1))
                if angle<np.radians(15): shadow+=1
            scores[i]=max(0.0, 1.0-shadow/5.0)
        return scores

    def _crossing_feature(self):
        """Fraction of edges that cross at least one other edge (per-node proxy)."""
        edges=self._get_unique_edges(); N=len(edges); cross_nodes=np.zeros(self.num_nodes)
        if N==0: return cross_nodes
        for k,(u1,v1) in enumerate(edges):
            l1=LineString([self.node_positions[u1],self.node_positions[v1]])
            for (u2,v2) in edges[k+1:]:
                if {u1,v1}&{u2,v2}: continue
                l2=LineString([self.node_positions[u2],self.node_positions[v2]])
                if l1.intersects(l2) and not l1.touches(l2):
                    for n in [u1,v1,u2,v2]: cross_nodes[n]+=0.25
        return cross_nodes/max(N,1)

    def _add_edge(self, u, v):
        if self.adjacency_matrix[u,v]>0: return False
        self.adjacency_matrix[u,v]=self.adjacency_matrix[v,u]=1
        dist=np.linalg.norm(self.node_positions[u]-self.node_positions[v])
        res=EnvironmentConfig.CABLE_RESISTANCE_PER_METER*dist
        W,H=EnvironmentConfig.GRID_SIZE; grid_diag=math.sqrt(W**2+H**2)
        new_e   =torch.tensor([[u,v],[v,u]], dtype=torch.long)
        new_attr=torch.tensor([[dist/grid_diag, res/(EnvironmentConfig.CABLE_RESISTANCE_PER_METER*grid_diag)],
                                [dist/grid_diag, res/(EnvironmentConfig.CABLE_RESISTANCE_PER_METER*grid_diag)]],
                               dtype=torch.float32)
        self.edge_index=torch.cat([self.edge_index,new_e],dim=1)
        self.edge_attr =torch.cat([self.edge_attr, new_attr],dim=0)
        return True

    def _remove_edge(self, u, v):
        if self.adjacency_matrix[u,v]==0: return False
        self.adjacency_matrix[u,v]=self.adjacency_matrix[v,u]=0
        W,H=EnvironmentConfig.GRID_SIZE; grid_diag=math.sqrt(W**2+H**2)
        rows,cols=np.where(self.adjacency_matrix>0)
        new_edges=[]; new_attrs=[]
        for r,c in zip(rows,cols):
            d=np.linalg.norm(self.node_positions[r]-self.node_positions[c])
            res=EnvironmentConfig.CABLE_RESISTANCE_PER_METER*d
            new_edges.append([r,c])
            new_attrs.append([d/grid_diag, res/(EnvironmentConfig.CABLE_RESISTANCE_PER_METER*grid_diag)])
        if new_edges:
            self.edge_index=torch.tensor(new_edges,dtype=torch.long).t()
            self.edge_attr =torch.tensor(new_attrs,dtype=torch.float32)
        else:
            self.edge_index=torch.empty((2,0),dtype=torch.long)
            self.edge_attr =torch.empty((0,GNNConfig.EDGE_FEATURE_DIM-1),dtype=torch.float32)
        return True


    def _violates_obstacles(self, line):
        for zone in EnvironmentConfig.FORBIDDEN_ZONES:
            poly=Point(zone['center']).buffer(zone['radius'])
            if line.intersects(poly) and not line.touches(poly):
                return True
        return False

    def _count_crossings(self, new_line, exclude_pair=None):
        count=0
        for u,v in self._get_unique_edges():
            if exclude_pair and {u,v}==set(exclude_pair): continue
            existing=LineString([self.node_positions[u],self.node_positions[v]])
            if new_line.intersects(existing) and not new_line.touches(existing): count+=1
        return count


    def step(self, action):
        yaw_angles = None
        if isinstance(action, dict):
            yaw_angles = action.get('yaw_angles')
            action = action.get('topology')
        elif isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], tuple):
            action, yaw_angles = action
        self._apply_yaw_action(yaw_angles)
        u, v, act_type = action
        if act_type==1:
            line=LineString([self.node_positions[u],self.node_positions[v]])
            if self._violates_obstacles(line):
                self._update_node_features()
                return self._get_state(), -30.0, False, {'obstacle_rejection': True}
        prev_iso=self._count_isolated_turbines()
        crossings=0
        if act_type==1:
            line=LineString([self.node_positions[u],self.node_positions[v]])
            crossings=self._count_crossings(line)
            self._add_edge(u,v)
        else:
            self._remove_edge(u,v)
        self._update_node_features()
        self.current_step += 1
        curr_iso=self._count_isolated_turbines()
        done=self.current_step>=self.max_steps
        if curr_iso==0 and self.current_step>int(self.max_steps*0.6): done=True
        reward=self._compute_reward(prev_iso, curr_iso, done, crossings)
        self.prev_isolated_count=curr_iso
        info={}
        if done: info['final_isolated']=curr_iso
        return self._get_state(), reward, done, info


    def _compute_reward(self, prev_iso, curr_iso, done, crossings=0):
        """
        Reward components:
          r_milestone : 500 on first connectivity; 100 while connected
          r_delta     : 20 per newly-connected turbine
          r_terminal  : +500 if iso=0 at done; −20×iso² otherwise
          r_cost      : lifecycle cost improvement bonus (stage-gated)
          r_topo      : cable efficiency vs MST
          r_cross     : −40 per cable crossing
          r_progress  : [v3-8] progressive cost improvement vs rolling window best
          r_step      : −1 time pressure
        """
        r_milestone = 500.0 if (curr_iso==0 and prev_iso>0) else (100.0 if curr_iso==0 else 0.0)
        r_delta     = (prev_iso-curr_iso)*20.0
        r_cost=0.0; r_progress=0.0
        if curr_iso==0 and self.cost_weight>0:
            ci=self.calculate_lifecycle_cost(); cost_m=ci['total']/1e6
            # [v4-2] Dynamic cost reward: z-score relative to running cost distribution
            if self._cost_ema is None:
                self._cost_ema = cost_m
            else:
                alpha = 0.05
                self._cost_ema    = (1-alpha)*self._cost_ema + alpha*cost_m
                self._cost_std_ema= (1-alpha)*self._cost_std_ema + alpha*abs(cost_m - self._cost_ema)
            std = max(self._cost_std_ema, 2.0)
            z_score = (self._cost_ema - cost_m) / std   # positive when below-average cost
            r_cost  = float(np.clip(z_score * 80.0 * self.cost_weight, -150.0, 300.0))
            # [v3-8] Progressive improvement bonus vs rolling window
            if self.best_cost_window:
                win_best=min(self.best_cost_window)
                if cost_m < win_best-0.1:
                    r_progress=min((win_best-cost_m)*5.0*self.cost_weight, 200.0)

        r_topo=0.0
        if curr_iso==0:
            total_len=self._total_cable_length(); mst_len=self._mst_cable_length()
            if total_len>0:
                eff=mst_len/total_len; r_topo=max(0.0,(eff-0.85)*100.0)

        r_control=0.0
        if curr_iso==0 and self.cost_weight>0:
            cm=self.calculate_control_metrics()
            fatigue_term=(cm['fatigue_cost_saving']/1e6)*70.0
            power_term=cm['gross_power_gain_mw']*8.0
            yaw_cost_term=(cm['yaw_actuation_cost']/1e6)*45.0
            ti_safety_term=max(0.0, 0.15-cm['max_ti_yaw'])*80.0
            r_control=float(np.clip((fatigue_term+power_term-yaw_cost_term+ti_safety_term)*self.cost_weight,
                                    -150.0, 250.0))

        r_reliability=0.0
        if curr_iso==0 and self.cost_weight>0 and ReliabilityConfig.ENABLED:
            rm=self.calculate_reliability_metrics()
            redundancy_bonus=rm['redundancy_ratio']*ReliabilityConfig.REDUNDANCY_REWARD_WEIGHT
            reliable_bonus=160.0 if self._is_reliable_topology(rm) else 0.0
            n1_excess=max(0, rm['max_single_edge_outage_turbines']
                          - ReliabilityConfig.RELIABLE_MAX_N1_OUTAGE_TURBINES)
            red_shortfall=max(0.0, ReliabilityConfig.RELIABLE_MIN_REDUNDANCY_RATIO
                              - rm['redundancy_ratio'])
            ens_excess=max(0.0, rm['expected_unserved_energy_mwh']
                           - ReliabilityConfig.RELIABLE_MAX_ENS_MWH)
            n1_penalty=n1_excess*18.0
            red_penalty=red_shortfall*180.0
            outage_penalty=(ens_excess/1000.0)*15.0
            r_reliability=float(np.clip((redundancy_bonus+reliable_bonus
                                         -n1_penalty-red_penalty-outage_penalty)*self.cost_weight,
                                        -220.0, 320.0))

        r_terminal=0.0
        if done:
            r_terminal = 500.0 if curr_iso==0 else -(curr_iso**2)*20.0
            if curr_iso==0 and self.cost_weight>0:
                ci=self.calculate_lifecycle_cost(); cost_m=ci['total']/1e6
                self.best_cost_window.append(cost_m)

        r_cross = crossings*ConstraintConfig.CROSSING_PENALTY  # [v3-4] −40/crossing
        r_step  = -1.0

        total=(r_milestone+r_delta+r_cost+r_topo+r_terminal+r_cross+r_progress
               +r_control+r_reliability+r_step)
        return float(np.clip(total, -500.0, 1000.0))


    def _fatigue_economic_metrics(self, ti_eff, yaw_angles=None):
        ti_arr=np.asarray(ti_eff, dtype=float)
        ref=max(LifecycleConfig.FATIGUE_REFERENCE_TI, 1e-6)
        damage=(np.maximum(ti_arr, 0.0)/ref)**LifecycleConfig.FATIGUE_DAMAGE_EXPONENT
        damage_index=float(np.mean(damage)) if damage.size else 0.0
        critical_exceedance=float(np.mean(ti_arr>LifecycleConfig.FATIGUE_CRITICAL_TI)) if ti_arr.size else 0.0
        fatigue_cost=damage_index*LifecycleConfig.FATIGUE_COST_PER_DAMAGE_INDEX
        yaw_cost=0.0
        if yaw_angles is not None:
            mean_abs=float(np.mean(np.abs(yaw_angles))) if len(yaw_angles)>0 else 0.0
            yaw_cost=(mean_abs*LifecycleConfig.YAW_ACTUATION_COST_PER_DEG_YEAR
                      *LifecycleConfig.LIFESPAN)
        return {'fatigue_damage_index': damage_index,
                'fatigue_cost': fatigue_cost,
                'critical_ti_fraction': critical_exceedance,
                'max_ti': float(np.max(ti_arr)) if ti_arr.size else 0.0,
                'mean_ti': float(np.mean(ti_arr)) if ti_arr.size else 0.0,
                'yaw_actuation_cost': yaw_cost}


    def _power_from_effective_velocity(self, u_eff, yaw_angles, connected=None):
        R=EnvironmentConfig.ROTOR_DIAMETER/2.0
        yaw_pf=self._cfd_model._yaw_power_factor(yaw_angles)
        hc=(EnvironmentConfig.Z_HUB/EnvironmentConfig.Z_REF)**EnvironmentConfig.ABL_ALPHA
        total=0.0
        connected = set(range(self.num_turbines)) if connected is None else set(connected)
        for i in range(self.num_turbines):
            if i not in connected: continue
            p=(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2
               *(u_eff[i]*hc)**3*EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR*yaw_pf[i]/1e6)
            total += min(p, EnvironmentConfig.RATED_POWER)
        return float(total)


    def calculate_robust_metrics(self, nominal_net_power=None, nominal_fatigue=None):
        if not RobustConfig.ENABLED:
            return {'c_robust': 0.0}
        connected=[i for i in range(self.num_turbines) if i in self._get_connected_nodes()]
        if nominal_net_power is None:
            nominal_net_power=self.wake_cache.get('total_power', 0.0)
        if nominal_fatigue is None:
            nominal_fatigue=self._fatigue_economic_metrics(self.wake_cache['ti_effective'], self.yaw_angles)

        powers=[]; fatigue_costs=[]; max_tis=[]; weights=[]
        base=np.arange(self.num_turbines, dtype=float)
        for scen in RobustConfig.SCENARIOS:
            yaw_err=float(scen['dyaw'])*np.sin(1.7*base+0.3)
            yaw_s=np.clip(self.yaw_angles+yaw_err,
                          -EnvironmentConfig.YAW_MAX_DEG, EnvironmentConfig.YAW_MAX_DEG)
            ws=max(3.0, self.wind_speed+float(scen['du']))
            wd=self.wind_direction+float(scen['ddir'])
            _, u_s, ti_s = self._cfd_model.compute_power_with_ti(
                self.node_positions[:self.num_turbines,0],
                self.node_positions[:self.num_turbines,1],
                wind_speed=ws, wind_dir_deg=wd, yaw_angles=yaw_s)
            powers.append(self._power_from_effective_velocity(u_s, yaw_s, connected))
            fat_s=self._fatigue_economic_metrics(ti_s, yaw_s)
            fatigue_costs.append(fat_s['fatigue_cost'])
            max_tis.append(fat_s['max_ti'])
            weights.append(float(scen['weight']))

        weights=np.asarray(weights, dtype=float)
        weights=weights/max(np.sum(weights), 1e-9)
        powers=np.asarray(powers, dtype=float)
        fatigue_costs=np.asarray(fatigue_costs, dtype=float)
        max_tis=np.asarray(max_tis, dtype=float)
        expected_power=float(np.sum(weights*powers))
        expected_fatigue=float(np.sum(weights*fatigue_costs))
        order=np.argsort(powers)
        cutoff=max(1, int(math.ceil(len(powers)*RobustConfig.CVAR_ALPHA)))
        cvar_power=float(np.mean(powers[order[:cutoff]]))
        shortfall=max(0.0, float(nominal_net_power)-cvar_power)
        c_power_shortfall=(shortfall*LifecycleConfig.OPERATING_HOURS*LifecycleConfig.LIFESPAN
                           *LifecycleConfig.ENERGY_PRICE*RobustConfig.POWER_SHORTFALL_PRICE_MULT)
        c_ti_cvar=max(0.0, expected_fatigue-float(nominal_fatigue['fatigue_cost']))*RobustConfig.TI_CVAR_COST_MULT
        return {'expected_power_mw': expected_power,
                'cvar_power_mw': cvar_power,
                'worst_power_mw': float(np.min(powers)),
                'robust_power_shortfall_mw': shortfall,
                'expected_fatigue_cost': expected_fatigue,
                'nominal_fatigue_cost': float(nominal_fatigue['fatigue_cost']),
                'expected_max_ti': float(np.sum(weights*max_tis)),
                'worst_max_ti': float(np.max(max_tis)),
                'c_robust_power': c_power_shortfall,
                'c_robust_ti': c_ti_cvar,
                'c_robust': c_power_shortfall+c_ti_cvar}


    def calculate_reliability_metrics(self):
        if not ReliabilityConfig.ENABLED:
            return {'c_reliability': 0.0, 'redundancy_ratio': 0.0,
                    'max_single_edge_outage_turbines': 0,
                    'expected_unserved_energy_mwh': 0.0}
        edges=self._get_unique_edges()
        G=nx.Graph(); G.add_nodes_from(range(self.num_nodes))
        for u,v in edges: G.add_edge(u,v)
        sub_idx=self.num_turbines
        if not nx.is_connected(G):
            return {'c_reliability': 1e9, 'redundancy_ratio': 0.0,
                    'max_single_edge_outage_turbines': self.num_turbines,
                    'expected_unserved_energy_mwh': 1e9,
                    'n_bridge_edges': len(edges), 'n_cycle_edges': 0}

        if not self.wake_cache['is_valid']: self._precompute_wake()
        turbine_power=np.asarray(self.wake_cache.get('turbine_power', np.zeros(self.num_turbines)), dtype=float)
        expected_unserved=0.0
        max_outage=0; bridge_edges=0; cycle_edges=0
        for u,v in edges:
            d_km=float(np.linalg.norm(self.node_positions[u]-self.node_positions[v]))/1000.0
            H=G.copy(); H.remove_edge(u,v)
            if nx.is_connected(H):
                cycle_edges += 1
                continue
            bridge_edges += 1
            reachable=nx.node_connected_component(H, sub_idx) if H.has_node(sub_idx) else set()
            lost=[i for i in range(self.num_turbines) if i not in reachable]
            max_outage=max(max_outage, len(lost))
            lost_power=float(np.sum(turbine_power[lost])) if lost else 0.0
            failure_rate=d_km*ReliabilityConfig.CABLE_FAILURE_RATE_PER_KM_YEAR
            expected_unserved += failure_rate*ReliabilityConfig.REPAIR_TIME_HOURS*lost_power

        c_unserved=(expected_unserved*LifecycleConfig.ENERGY_PRICE
                    *ReliabilityConfig.UNSERVED_ENERGY_PRICE_MULT)
        n1_cost=(max_outage*ReliabilityConfig.N1_OUTAGE_TURBINE_PENALTY
                 *LifecycleConfig.FATIGUE_COST_PER_DAMAGE_INDEX)
        redundancy_ratio=cycle_edges/max(len(edges),1)
        return {'c_reliability': c_unserved+n1_cost,
                'c_unserved_energy': c_unserved,
                'c_n1_exposure': n1_cost,
                'expected_unserved_energy_mwh': expected_unserved,
                'max_single_edge_outage_turbines': int(max_outage),
                'n_bridge_edges': int(bridge_edges),
                'n_cycle_edges': int(cycle_edges),
                'redundancy_ratio': float(redundancy_ratio)}


    def _is_reliable_topology(self, rel_metrics):
        return (
            rel_metrics.get('max_single_edge_outage_turbines', self.num_turbines)
            <= ReliabilityConfig.RELIABLE_MAX_N1_OUTAGE_TURBINES
            and rel_metrics.get('redundancy_ratio', 0.0)
            >= ReliabilityConfig.RELIABLE_MIN_REDUNDANCY_RATIO
            and rel_metrics.get('expected_unserved_energy_mwh', float('inf'))
            <= ReliabilityConfig.RELIABLE_MAX_ENS_MWH
        )


    def calculate_control_metrics(self):
        if not self.wake_cache['is_valid']: self._precompute_wake()
        zero_yaw=np.zeros(self.num_turbines, dtype=float)
        _, u_zero, ti_zero = self._cfd_model.compute_power_with_ti(
            self.node_positions[:self.num_turbines,0],
            self.node_positions[:self.num_turbines,1],
            wind_speed=self.wind_speed,
            wind_dir_deg=self.wind_direction,
            yaw_angles=zero_yaw,
        )
        R=EnvironmentConfig.ROTOR_DIAMETER/2.0
        p_zero=np.array([
            min(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2
                *u_zero[i]**3*EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR/1e6,
                EnvironmentConfig.RATED_POWER)
            for i in range(self.num_turbines)
        ], dtype=float)
        p_yaw=self.wake_cache.get('turbine_power', np.zeros(self.num_turbines))
        ti_yaw=self.wake_cache['ti_effective']
        yaw_effort=float(np.mean(np.abs(self.yaw_angles))/EnvironmentConfig.YAW_MAX_DEG)
        ti_load_zero=float(np.mean(ti_zero**2))
        ti_load_yaw=float(np.mean(ti_yaw**2))
        fatigue_zero=self._fatigue_economic_metrics(ti_zero, zero_yaw)
        fatigue_yaw=self._fatigue_economic_metrics(ti_yaw, self.yaw_angles)
        return {
            'mean_abs_yaw_deg': float(np.mean(np.abs(self.yaw_angles))),
            'max_abs_yaw_deg': float(np.max(np.abs(self.yaw_angles))) if self.num_turbines else 0.0,
            'yaw_effort_raw': yaw_effort,
            'yaw_effort': yaw_effort*EnvironmentConfig.YAW_EFFORT_PENALTY,
            'gross_power_zero_yaw_mw': float(np.sum(p_zero)),
            'gross_power_yaw_mw': float(np.sum(p_yaw)),
            'gross_power_gain_mw': float(np.sum(p_yaw)-np.sum(p_zero)),
            'ti_load_zero_yaw': ti_load_zero,
            'ti_load_yaw': ti_load_yaw,
            'ti_load_reduction': ti_load_zero-ti_load_yaw,
            'mean_ti_zero_yaw': float(np.mean(ti_zero)),
            'mean_ti_yaw': float(np.mean(ti_yaw)),
            'max_ti_zero_yaw': fatigue_zero['max_ti'],
            'max_ti_yaw': fatigue_yaw['max_ti'],
            'fatigue_damage_zero_yaw': fatigue_zero['fatigue_damage_index'],
            'fatigue_damage_yaw': fatigue_yaw['fatigue_damage_index'],
            'fatigue_damage_reduction': fatigue_zero['fatigue_damage_index']-fatigue_yaw['fatigue_damage_index'],
            'fatigue_cost_zero_yaw': fatigue_zero['fatigue_cost'],
            'fatigue_cost_yaw': fatigue_yaw['fatigue_cost'],
            'fatigue_cost_saving': fatigue_zero['fatigue_cost']-fatigue_yaw['fatigue_cost'],
            'yaw_actuation_cost': fatigue_yaw['yaw_actuation_cost'],
            'critical_ti_fraction_yaw': fatigue_yaw['critical_ti_fraction'],
        }


    def calculate_lifecycle_cost(self):
        if not self.wake_cache['is_valid']: self._precompute_wake()
        total_len_m=self._total_cable_length()
        c_inv=(total_len_m/1000.0)*LifecycleConfig.CABLE_COST_PER_KM
        c_op=0.5*c_inv
        connected=self._get_connected_nodes()
        R=EnvironmentConfig.ROTOR_DIAMETER/2.0
        eff_vels=self.wake_cache['effective_velocities']
        yaw_pf=self.wake_cache.get('yaw_power_factor', np.ones(self.num_turbines))
        hc=(EnvironmentConfig.Z_HUB/EnvironmentConfig.Z_REF)**EnvironmentConfig.ABL_ALPHA
        deliv_power=0.0
        for i in range(self.num_turbines):
            if i in connected:
                vc=eff_vels[i]*hc
                p=(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2
                   *vc**3*EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR*yaw_pf[i]/1e6)
                deliv_power+=min(p, EnvironmentConfig.RATED_POWER)
        ohmic_mw=self._ohmic_loss()
        net_power=max(0.0, deliv_power-ohmic_mw)
        baseline_mw=self.num_turbines*EnvironmentConfig.RATED_POWER
        lost_mw=baseline_mw-net_power
        c_wake=(lost_mw*LifecycleConfig.OPERATING_HOURS*LifecycleConfig.LIFESPAN
                *LifecycleConfig.ENERGY_PRICE)
        fatigue=self._fatigue_economic_metrics(self.wake_cache['ti_effective'], self.yaw_angles)
        c_fatigue=fatigue['fatigue_cost']
        c_yaw_act=fatigue['yaw_actuation_cost']
        c_fatigue_route=self._fatigue_route_cost()
        robust=self.calculate_robust_metrics(nominal_net_power=net_power, nominal_fatigue=fatigue)
        c_robust=robust.get('c_robust', 0.0)
        c_robust_route=self._robust_route_cost()
        reliability=self.calculate_reliability_metrics()
        c_reliability=reliability.get('c_reliability', 0.0)
        n_connected=sum(1 for i in range(self.num_turbines) if i in connected)
        return {'total': c_inv+c_op+c_wake+c_fatigue+c_yaw_act+c_fatigue_route+c_robust+c_robust_route+c_reliability,
                'c_inv': c_inv, 'c_op': c_op, 'c_wake': c_wake,
                'c_fatigue': c_fatigue, 'c_yaw_act': c_yaw_act,
                'c_fatigue_route': c_fatigue_route,
                'c_robust': c_robust, 'c_robust_route': c_robust_route,
                'c_reliability': c_reliability,
                'c_unserved_energy': reliability.get('c_unserved_energy', 0.0),
                'c_n1_exposure': reliability.get('c_n1_exposure', 0.0),
                'expected_unserved_energy_mwh': reliability.get('expected_unserved_energy_mwh', 0.0),
                'max_single_edge_outage_turbines': reliability.get('max_single_edge_outage_turbines', 0),
                'n_bridge_edges': reliability.get('n_bridge_edges', 0),
                'n_cycle_edges': reliability.get('n_cycle_edges', 0),
                'redundancy_ratio': reliability.get('redundancy_ratio', 0.0),
                'c_robust_power': robust.get('c_robust_power', 0.0),
                'c_robust_ti': robust.get('c_robust_ti', 0.0),
                'expected_power_mw': robust.get('expected_power_mw', net_power),
                'cvar_power_mw': robust.get('cvar_power_mw', net_power),
                'worst_power_mw': robust.get('worst_power_mw', net_power),
                'expected_max_ti': robust.get('expected_max_ti', fatigue['max_ti']),
                'worst_max_ti': robust.get('worst_max_ti', fatigue['max_ti']),
                'fatigue_damage_index': fatigue['fatigue_damage_index'],
                'critical_ti_fraction': fatigue['critical_ti_fraction'],
                'max_ti_eff': fatigue['max_ti'],
                'ohmic_loss': ohmic_mw, 'deliverable_power': deliv_power, 'net_power': net_power,
                'connected_turbines': n_connected}

    def _electrical_path_lengths_km(self):
        G=nx.Graph(); rows,cols=np.where(self.adjacency_matrix>0)
        for r,c in zip(rows,cols):
            if r<c:
                d=float(np.linalg.norm(self.node_positions[r]-self.node_positions[c]))
                G.add_edge(r,c, weight=d)
        sub_idx=self.num_turbines
        out=np.full(self.num_turbines, np.inf, dtype=float)
        if not G.has_node(sub_idx): return out
        try:
            lengths=nx.single_source_dijkstra_path_length(G, sub_idx, weight='weight')
            for i in range(self.num_turbines):
                if i in lengths: out[i]=lengths[i]/1000.0
        except Exception:
            pass
        return out


    def _fatigue_route_cost(self):
        if not self.wake_cache['is_valid']: self._precompute_wake()
        path_km=self._electrical_path_lengths_km()
        if np.any(~np.isfinite(path_km)):
            return 1e9
        ti=np.asarray(self.wake_cache['ti_effective'], dtype=float)
        ref=max(LifecycleConfig.FATIGUE_REFERENCE_TI, 1e-6)
        damage_w=(np.maximum(ti, 0.0)/ref)**LifecycleConfig.FATIGUE_DAMAGE_EXPONENT
        weighted_path=float(np.sum(damage_w*path_km)/max(np.sum(damage_w), 1e-9))
        critical=ti>LifecycleConfig.FATIGUE_CRITICAL_TI
        critical_path=float(np.mean(path_km[critical])) if np.any(critical) else 0.0
        return (weighted_path*LifecycleConfig.FATIGUE_ROUTE_COST_PER_WEIGHTED_KM
                + critical_path*LifecycleConfig.FATIGUE_CRITICAL_ROUTE_COST_PER_KM)


    def _robust_route_cost(self):
        if not self.wake_cache['is_valid']: self._precompute_wake()
        path_km=self._electrical_path_lengths_km()
        if np.any(~np.isfinite(path_km)):
            return 1e9
        power=np.asarray(self.wake_cache.get('turbine_power', np.zeros(self.num_turbines)), dtype=float)
        ti=np.asarray(self.wake_cache['ti_effective'], dtype=float)
        power_shortfall=np.maximum(0.0, EnvironmentConfig.RATED_POWER-power)/max(EnvironmentConfig.RATED_POWER,1e-6)
        ti_excess=np.maximum(0.0, ti-LifecycleConfig.FATIGUE_REFERENCE_TI)/max(LifecycleConfig.FATIGUE_REFERENCE_TI,1e-6)
        vulnerability=1.0+power_shortfall+0.65*ti_excess
        weighted_path=float(np.sum(vulnerability*path_km)/max(np.sum(vulnerability),1e-9))
        low_power=power<np.percentile(power, 35.0)
        low_power_path=float(np.mean(path_km[low_power])) if np.any(low_power) else 0.0
        return (weighted_path*RobustConfig.ROBUST_ROUTE_COST_PER_WEIGHTED_KM
                + low_power_path*RobustConfig.ROBUST_LOW_POWER_ROUTE_COST_PER_KM)


    def _ohmic_loss(self):
        G=nx.Graph(); rows,cols=np.where(self.adjacency_matrix>0)
        for r,c in zip(rows,cols):
            if r<c:
                d=np.linalg.norm(self.node_positions[r]-self.node_positions[c])
                G.add_edge(r,c, weight=d)
        if not self.wake_cache['is_valid']: self._precompute_wake()
        R=EnvironmentConfig.ROTOR_DIAMETER/2.0; eff_v=self.wake_cache['effective_velocities']
        yaw_pf=self.wake_cache.get('yaw_power_factor', np.ones(self.num_turbines))
        powers=np.zeros(self.num_nodes)
        for i in range(self.num_turbines):
            p_mw=(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2
                  *eff_v[i]**3*EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR*yaw_pf[i]/1e6)
            powers[i]=min(p_mw, EnvironmentConfig.RATED_POWER)*1e6
        sub_idx=self.num_turbines
        if not G.has_node(sub_idx): return 0.0
        try:
            paths=nx.shortest_path(G, target=sub_idx, weight='weight')
            currents={}
            for i in range(self.num_turbines):
                if i not in paths: continue
                path=paths[i]
                I=powers[i]/(math.sqrt(3)*EnvironmentConfig.COLLECTION_VOLTAGE
                              *1000*EnvironmentConfig.POWER_FACTOR)
                for k in range(len(path)-1):
                    e=tuple(sorted((path[k],path[k+1]))); currents[e]=currents.get(e,0.0)+I
            loss_w=0.0
            for (u,v),I in currents.items():
                d=np.linalg.norm(self.node_positions[u]-self.node_positions[v])
                loss_w+=I**2*(d*EnvironmentConfig.CABLE_RESISTANCE_PER_METER)
            return loss_w/1e6
        except Exception: return 0.0

    def _total_cable_length(self):
        length=0.0; rows,cols=np.where(self.adjacency_matrix>0)
        for r,c in zip(rows,cols):
            if r<c: length+=np.linalg.norm(self.node_positions[r]-self.node_positions[c])
        return length

    def _mst_cable_length(self):
        G=nx.Graph()
        for i in range(self.num_nodes):
            for j in range(i+1,self.num_nodes):
                d=np.linalg.norm(self.node_positions[i]-self.node_positions[j])
                if d<=EnvironmentConfig.MAX_CABLE_LENGTH: G.add_edge(i,j,weight=d)
        if G.number_of_nodes()<self.num_nodes: return self._total_cable_length()
        try:
            mst=nx.minimum_spanning_tree(G)
            return sum(d['weight'] for _,_,d in mst.edges(data=True))
        except Exception: return self._total_cable_length()


    def validate_topology(self):
        return _base_validate(self)


# ============================================================================
# SECTION 8: NSGA-II LAYOUT OPTIMISER  [Innovation I2]
# ============================================================================

BASE_REQUIRE_N1 = True

# Embedded in each standalone baseline script; no extra server-side module is needed.
BASELINE_REVISION = 'base25-planar-cut-feasibility-20261016'


class BaselineFeasibilityConfig:
    MILP_TIME_LIMIT_SECONDS = 120.0
    MILP_ROUND_SECONDS = 10.0
    MILP_MAX_ROUNDS = 64


def _base_snapshot(env, tag):
    return {'adj_matrix': env.adjacency_matrix.copy(),
            'edge_index': env.edge_index.clone(), 'edge_attr': env.edge_attr.clone(),
            'yaw_angles': env.yaw_angles.copy(),
            'cost_m': float(env.calculate_lifecycle_cost()['total']) / 1e6,
            'episode': -1, 'validation': env.validate_topology(), 'tag': tag}


def _base_restore(env, state):
    # Never lend a saved array to a mutable environment.
    env.adjacency_matrix = state['adj_matrix'].copy()
    env.edge_index = state['edge_index'].clone()
    env.edge_attr = state['edge_attr'].clone()
    yaw = np.asarray(state['yaw_angles'], dtype=float)
    if not np.array_equal(env.yaw_angles, yaw):
        env.wake_cache['is_valid'] = False
    env.yaw_angles = yaw.copy()
    env._update_node_features()


def _base_set_edges(env, edges):
    for u, v in sorted(env._get_unique_edges()):
        env._remove_edge(u, v)
    for u, v in sorted(edges):
        env._add_edge(int(u), int(v))
    env._update_node_features()


def _base_edge_conflict(positions, first, second):
    a = LineString([positions[i] for i in first])
    b = LineString([positions[i] for i in second])
    if set(first) & set(second):
        return a.intersection(b).length > 1e-7
    return a.intersects(b)


def _base_validate(env):
    positions = np.asarray(env.node_positions, dtype=float)
    finite = bool(np.isfinite(positions).all())
    graph = env._build_nx_graph()
    edges = sorted(graph.edges())
    connected = nx.is_connected(graph)
    reached = nx.node_connected_component(graph, env.num_turbines)
    isolated = sum(i not in reached for i in range(env.num_turbines))
    bridges = list(nx.bridges(graph))
    worst = isolated
    for u, v in bridges:
        graph.remove_edge(u, v)
        reached = nx.node_connected_component(graph, env.num_turbines)
        worst = max(worst, sum(i not in reached for i in range(env.num_turbines)))
        graph.add_edge(u, v)
    layout_ok = finite and HybridAICFDLayoutOptimizer._layout_violation_score(
        type('LayoutContext', (), {'num_turbines': env.num_turbines})(),
        positions[:env.num_turbines]) <= 1e-6
    obstacle = length = crossings = 0
    if finite:
        for u, v in edges:
            distance = float(np.linalg.norm(positions[u] - positions[v]))
            length += int(distance <= 1e-7 or distance > EnvironmentConfig.MAX_CABLE_LENGTH + 1e-7)
            obstacle += int(env._violates_obstacles(LineString([positions[u], positions[v]])))
        for i, edge in enumerate(edges):
            crossings += sum(_base_edge_conflict(positions, edge, other) for other in edges[i+1:])
    n1_secure = bool(connected and not bridges and worst == 0)
    geometry = bool(layout_ok and not obstacle and not length and crossings == 0)
    valid = bool(connected and geometry and (not BASE_REQUIRE_N1 or n1_secure))
    result = {'fully_connected': connected, 'isolated_count': int(isolated),
              'layout_feasible': bool(layout_ok), 'finite_positions': finite,
              'obstacle_violations': int(obstacle), 'cable_length_violations': int(length),
              'crossing_violations': int(crossings), 'geometry_valid': geometry,
              'n1_secure': n1_secure, 'n_bridge_edges': len(bridges),
              'max_single_edge_outage_turbines': int(worst), 'is_valid': valid,
              'feasibility_revision': BASELINE_REVISION}
    if BASE_REQUIRE_N1:
        rel = env.calculate_reliability_metrics()
        result.update(redundancy_ratio=float(rel.get('redundancy_ratio', 0)),
                      expected_unserved_energy_mwh=float(rel['expected_unserved_energy_mwh']))
    return result


def _base_graph_feasible(graph):
    return nx.is_connected(graph) and (not BASE_REQUIRE_N1 or not any(nx.bridges(graph)))


def _base_milp_planar_candidate(env, legal):
    """Find a legal planar graph using binary edges and connectivity cuts."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    started = time.perf_counter()
    positions = np.asarray(env.node_positions, dtype=float)
    edges = [(u, v) for _, u, v in legal]
    n_edges = len(edges)
    required_cut = 2 if BASE_REQUIRE_N1 else 1
    rows, cols, values, lower, upper = [], [], [], [], []
    cuts = set()

    def add_constraint(indices, lb, ub):
        row = len(lower)
        indices = list(indices)
        rows.extend([row] * len(indices))
        cols.extend(indices)
        values.extend([1.0] * len(indices))
        lower.append(lb)
        upper.append(ub)

    def add_cut(nodes):
        side = frozenset(nodes)
        complement = frozenset(range(env.num_nodes)) - side
        if not side or not complement:
            return False
        key = min(tuple(sorted(side)), tuple(sorted(complement)))
        if key in cuts:
            return False
        cuts.add(key)
        add_constraint((i for i, (u, v) in enumerate(edges)
                        if (u in side) != (v in side)), required_cut, np.inf)
        return True

    # Degree bounds alone permit disconnected cycles; cuts below separate them.
    for node in range(env.num_nodes):
        add_cut({node})
    conflicts = 0
    for i, edge in enumerate(edges):
        for j in range(i + 1, n_edges):
            if _base_edge_conflict(positions, edge, edges[j]):
                add_constraint((i, j), -np.inf, 1.0)
                conflicts += 1

    diagnostics = {'revision': BASELINE_REVISION, 'requires_n1': BASE_REQUIRE_N1,
                   'legal_edges': n_edges, 'conflict_pairs': conflicts,
                   'time_limit_s': BaselineFeasibilityConfig.MILP_TIME_LIMIT_SECONDS,
                   'rounds': [], 'status': 'SEARCH_LIMIT'}
    LOG.info('[BASE PREFLIGHT] greedy constructions exhausted; MILP edges=%d conflicts=%d',
             n_edges, conflicts)
    for iteration in range(BaselineFeasibilityConfig.MILP_MAX_ROUNDS):
        remaining = BaselineFeasibilityConfig.MILP_TIME_LIMIT_SECONDS - (time.perf_counter() - started)
        if remaining <= 0:
            break
        matrix = coo_matrix((values, (rows, cols)),
                            shape=(len(lower), n_edges)).tocsc()
        # This is a feasibility fallback, not a replacement cost optimizer.
        result = milp(c=np.zeros(n_edges), integrality=np.ones(n_edges),
                      bounds=Bounds(np.zeros(n_edges), np.ones(n_edges)),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={'time_limit': min(remaining, BaselineFeasibilityConfig.MILP_ROUND_SECONDS),
                               'presolve': True})
        record = {'round': iteration + 1, 'solver_status': int(result.status),
                  'message': str(result.message), 'cut_count': len(cuts)}
        diagnostics['rounds'].append(record)
        LOG.info('[BASE PREFLIGHT] MILP round=%d solver_status=%d cuts=%d elapsed=%.1fs',
                 iteration + 1, result.status, len(cuts), time.perf_counter() - started)
        if result.x is None:
            diagnostics['status'] = 'MODEL_INFEASIBLE' if result.status == 2 else 'NO_INCUMBENT'
            break
        x = np.asarray(result.x, dtype=float)
        if not np.isfinite(x).all() or np.max(np.abs(x - np.rint(x))) > 1e-5:
            diagnostics['status'] = 'NO_INTEGER_INCUMBENT'
            break
        graph = nx.Graph()
        graph.add_nodes_from(range(env.num_nodes))
        graph.add_edges_from(edge for edge, selected in zip(edges, x) if selected > 0.5)
        record.update(components=nx.number_connected_components(graph),
                      bridges=len(list(nx.bridges(graph))))
        if _base_graph_feasible(graph):
            # Removing edges preserves geometry; retain connectivity and N-1.
            for _, u, v in reversed(legal):
                if not graph.has_edge(u, v):
                    continue
                graph.remove_edge(u, v)
                if not _base_graph_feasible(graph):
                    graph.add_edge(u, v)
            diagnostics.update(status='FOUND', edges=graph.number_of_edges(),
                               elapsed_s=time.perf_counter() - started)
            save_json(diagnostics, 'baseline_planar_solver.json')
            return sorted(graph.edges())
        added = 0
        for component in nx.connected_components(graph):
            added += int(add_cut(component))
        if BASE_REQUIRE_N1:
            for u, v in list(nx.bridges(graph)):
                graph.remove_edge(u, v)
                added += int(add_cut(nx.node_connected_component(graph, u)))
                graph.add_edge(u, v)
        if not added:
            diagnostics['status'] = 'NO_NEW_CUTS'
            break
    diagnostics['elapsed_s'] = time.perf_counter() - started
    save_json(diagnostics, 'baseline_planar_solver.json')
    raise RuntimeError('Planar feasibility search did not produce a validated candidate; '
                       'see baseline_preflight_input.json and baseline_planar_solver.json '
                       '(status=%s).' % diagnostics['status'])


def _base_planar_candidates(env):
    from scipy.spatial import Delaunay, QhullError
    positions = np.asarray(env.node_positions, dtype=float)
    legal = []
    for u in range(env.num_nodes):
        for v in range(u+1, env.num_nodes):
            distance = float(np.linalg.norm(positions[u]-positions[v]))
            if (1e-7 < distance <= EnvironmentConfig.MAX_CABLE_LENGTH
                    and not env._violates_obstacles(LineString([positions[u], positions[v]]))):
                legal.append((distance, u, v))
    legal.sort()
    allowed = {(u,v) for _,u,v in legal}
    full = nx.Graph(); full.add_nodes_from(range(env.num_nodes)); full.add_edges_from(allowed)
    if not _base_graph_feasible(full):
        raise RuntimeError('Length/obstacle-legal graph cannot meet the connectivity requirement.')
    triangulated = set()
    try:
        for triangle in Delaunay(positions).simplices:
            for i in range(3):
                u,v = sorted((int(triangle[i]), int(triangle[(i+1)%3])))
                if (u,v) in allowed:
                    triangulated.add((u,v))
    except QhullError:
        pass
    for name, seed in [('delaunay', triangulated), ('shortest_planar', set())]:
        graph = nx.Graph(); graph.add_nodes_from(range(env.num_nodes))
        for u,v in sorted(seed):
            if not any(_base_edge_conflict(positions,(u,v),e) for e in graph.edges()):
                graph.add_edge(u,v,weight=float(np.linalg.norm(positions[u]-positions[v])))
        for distance,u,v in legal:
            if _base_graph_feasible(graph):
                break
            if graph.has_edge(u,v) or any(_base_edge_conflict(positions,(u,v),e) for e in graph.edges()):
                continue
            graph.add_edge(u,v,weight=distance)
        if _base_graph_feasible(graph):
            if not BASE_REQUIRE_N1:
                graph = nx.minimum_spanning_tree(graph)
            yield name, sorted(graph.edges())
        else:
            LOG.info('[BASE PREFLIGHT] construction=%s incomplete components=%d bridges=%d',
                     name, nx.number_connected_components(graph), len(list(nx.bridges(graph))))
    yield 'milp_planar_cuts', _base_milp_planar_candidate(env, legal)


def _base_ensure_feasible(env):
    before = _base_snapshot(env, 'before_feasibility_repair')
    if before['validation']['is_valid']:
        return 'already_feasible'
    try:
        if not before['validation']['layout_feasible']:
            raise RuntimeError('Invalid fixed layout; refuse to change positions after training.')
        for name, edges in _base_planar_candidates(env):
            _base_set_edges(env, edges)
            if env.validate_topology()['is_valid']:
                LOG.info('[BASE REPAIR] construction=%s edges=%d', name, len(edges))
                return name
        raise RuntimeError('No feasible planar construction found; this is not an infeasibility proof.')
    except Exception:
        _base_restore(env, before)
        raise


def _base_shorter_edge_refinement(env, max_iters=200):
    """Keep the baseline's shorter-edge search, with full-objective acceptance."""
    if not env.validate_topology()['is_valid']:
        raise RuntimeError('Refinement requires a feasible incumbent.')
    initial = _base_snapshot(env, 'before_shorter_edge_refinement')
    best_cost = initial['cost_m']
    accepted_count = 0
    positions = env.node_positions
    for iteration in range(max_iters):
        accepted = False
        ordered = sorted(env._get_unique_edges(), key=lambda e: (-np.linalg.norm(positions[e[0]]-positions[e[1]]),e))
        for u,v in ordered:
            base = _base_snapshot(env, 'trial_backup')
            old_length = float(np.linalg.norm(positions[u]-positions[v]))
            candidates = [None]
            candidates += [(i,j) for i in range(env.num_nodes) for j in range(i+1,env.num_nodes)
                           if env.adjacency_matrix[i,j] == 0
                           and 1e-7 < np.linalg.norm(positions[i]-positions[j]) < old_length-1e-7
                           and np.linalg.norm(positions[i]-positions[j]) <= EnvironmentConfig.MAX_CABLE_LENGTH
                           and not env._violates_obstacles(LineString([positions[i],positions[j]]))]
            candidates[1:] = sorted(candidates[1:],key=lambda e:(np.linalg.norm(positions[e[0]]-positions[e[1]]),e))
            for pair in candidates:
                _base_restore(env,base)
                env._remove_edge(u,v)
                if pair is not None:
                    env._add_edge(*pair)
                # Cheap graph screening avoids evaluating disconnected tentative layouts.
                if not _base_graph_feasible(env._build_nx_graph()):
                    continue
                if not env.validate_topology()['is_valid']:
                    continue
                cost = float(env.calculate_lifecycle_cost()['total'])/1e6
                if np.isfinite(cost) and cost < best_cost-1e-8:
                    best_cost=cost; accepted=True; accepted_count+=1
                    break
            if accepted:
                env._update_node_features()
                break
            _base_restore(env,base)
        if not accepted:
            break
        LOG.info('[BASE REFINE] iteration=%d accepted=%d score=%.6fM',iteration+1,accepted_count,best_cost)
    final = env.validate_topology()
    final_cost = float(env.calculate_lifecycle_cost()['total'])/1e6
    if not final['is_valid'] or not np.isfinite(final_cost) or final_cost > initial['cost_m']+1e-8:
        _base_restore(env,initial)
        raise RuntimeError('Refinement invariant violated; restored feasible incumbent.')
    return {'iters_improved':accepted_count,'initial_cost_m':initial['cost_m'],'final_cost_m':final_cost,
            'delta_cost_m':initial['cost_m']-final_cost,
            'initial_cable_km':float(sum(np.linalg.norm(positions[u]-positions[v])
                for u in range(env.num_nodes) for v in range(u+1,env.num_nodes) if initial['adj_matrix'][u,v]>0))/1000,
            'final_cable_km':env._total_cable_length()/1000,
            'final_validation':final,'objective':'aligned-full-score'}


def _base_preflight(env):
    initial = _base_snapshot(env,'preflight_input')
    save_json({'revision': BASELINE_REVISION, 'seed': RANDOM_SEED,
               'requires_n1': BASE_REQUIRE_N1,
               'node_positions': np.asarray(env.node_positions, dtype=float).tolist(),
               'num_turbines': env.num_turbines,
               'max_cable_length_m': EnvironmentConfig.MAX_CABLE_LENGTH,
               'forbidden_zones': EnvironmentConfig.FORBIDDEN_ZONES,
               'input_validation': initial['validation']}, 'baseline_preflight_input.json')
    try:
        method = _base_ensure_feasible(env)
        fallback = _base_snapshot(env,'deterministic_feasible_fallback')
        save_json({'status':'PASS','revision':BASELINE_REVISION,'method':method,
                   'validation':fallback['validation']},'baseline_preflight.json')
        return fallback
    except Exception as exc:
        save_json({'status':'FAILED_PREFLIGHT','revision':BASELINE_REVISION,'error':str(exc)},
                  'baseline_feasibility_status.json')
        raise
    finally:
        _base_restore(env,initial)


def _base_finalize(env, active_topo, fallback):
    selected = active_topo or fallback
    _base_restore(env,selected)
    selected_cost = float(env.calculate_lifecycle_cost()['total'])/1e6
    selected_length = env._total_cable_length()/1000
    fallback_used = active_topo is None or active_topo is fallback
    try:
        method = _base_ensure_feasible(env)
    except RuntimeError as exc:
        LOG.warning('[BASE FINAL] %s; using recorded feasible fallback.',exc)
        _base_restore(env,fallback)
        selected = fallback; fallback_used = True; method = 'preflight_fallback'
    feasible = _base_snapshot(env,'feasible_before_refinement')
    refinement = _base_shorter_edge_refinement(env,max_iters=200)
    final = env.validate_topology()
    if not final['is_valid'] or not np.isfinite(env.calculate_lifecycle_cost()['total']):
        _base_restore(env,feasible)
        raise RuntimeError('Final feasibility check failed; do not mark this run COMPLETE.')
    refinement.update(revision=BASELINE_REVISION,feasibility_repair=method,fallback_used=fallback_used,
                      selected_candidate_cost_m=selected_cost,selected_candidate_cable_km=selected_length,
                      repaired_feasible_cost_m=feasible['cost_m'],
                      final_edges=len(env._get_unique_edges()),
                      final_crossings=final['crossing_violations'],preserved_reliability=final['n1_secure'])
    save_json({'status':'FEASIBLE','revision':BASELINE_REVISION,'fallback_used':fallback_used,
               'repair_method':method,'validation':final},'baseline_feasibility_status.json')
    return selected, refinement


def _base_write_metadata():
    import hashlib
    import platform
    from pathlib import Path
    path = Path(RESULTS_DIR) / 'run_metadata.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata.update(baseline_revision=BASELINE_REVISION,
                    objective_definition='aligned-routing-score',
                    algorithm='single-layout-search-baseline',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python_version=platform.python_version(), pytorch_version=torch.__version__,
                    cuda_version=torch.version.cuda, requires_n1=BASE_REQUIRE_N1)
    metadata['configuration'] = {
        name: {key:value for key,value in vars(cls).items() if key.isupper()}
        for name,cls in globals().items() if isinstance(cls,type) and name.endswith('Config')}
    path.write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')


class HybridAICFDLayoutOptimizer:
    """
    Multi-objective NSGA-II  [Innovation I2].
    Objectives: AEP↑, wake-loss ratio↓, structural TI-load index↓, cable cost↓.
    Crowding-distance secondary sorting for Pareto diversity.
    Load index now uses Frandsen TI standard deviation.
    """
    def __init__(self, env):
        self.env=env; self.num_turbines=env.num_turbines
        self.grid_size=np.array(EnvironmentConfig.GRID_SIZE)
        self.R=EnvironmentConfig.ROTOR_DIAMETER/2.0
        self.scenarios=self._build_climate_scenarios()
        self.cfd_model=EnhancedCFDModel()
        self.final_power=0.0; self.final_wake_ratio=0.0
        self.final_load_index=0.0; self.cable_cost=0.0

    def _build_climate_scenarios(self):
        weibull={8.0:0.15, 10.0:0.25, 12.0:0.35, 14.0:0.25}
        dirs={-15:0.25, 0:0.50, 15:0.25}
        return [{'wind_speed':ws,'wind_direction':wd,'weight':ww*dw}
                for ws,ww in weibull.items() for wd,dw in dirs.items()]

    def optimize_layout(self):
        log_section("Phase 1 – NSGA-II Multi-Objective Layout Optimisation  [I2]")
        LOG.info(f"  Scenarios={len(self.scenarios)}  "
                 f"Pop={DataGenerationConfig.LAYOUT_CANDIDATES}  "
                 f"Gens={DataGenerationConfig.NSGA_GENERATIONS}")
        return self._nsga2(pop_size=DataGenerationConfig.LAYOUT_CANDIDATES,
                           generations=DataGenerationConfig.NSGA_GENERATIONS)

    def _nsga2(self, pop_size, generations):
        population=[self._rand_feasible() for _ in range(pop_size)]
        nsga_log=[]; best_layout=population[0]

        for gen in range(generations):
            objs=[]; valid=[]
            for layout in population:
                res=self._eval_objectives(layout)
                objs.append(res[:4]); valid.append(res[4])
            vi=[i for i,f in enumerate(valid) if f]
            if not vi:
                LOG.warning(f"  Gen {gen}: no feasible – reinitialising")
                population=[self._rand_feasible() for _ in range(pop_size)]; continue

            v_objs=[objs[i] for i in vi]
            fronts=self._fast_non_dom_sort(v_objs)
            mfronts=[[vi[i] for i in f] for f in fronts]

            # [v3-7] Crowding distance for diversity
            cd=self._crowding_distance(v_objs, fronts)
            cd_map={vi[i]: cd[i] for i in range(len(vi))}

            rank_map={}
            for r,front in enumerate(mfronts):
                for i in front: rank_map[i]=r

            def get_key(idx):
                r=rank_map.get(idx,999) if valid[idx] else 9999
                return (r, -cd_map.get(idx,0.0))

            if mfronts and mfronts[0]: best_layout=population[mfronts[0][0]]

            offspring=[]
            while len(offspring)<pop_size:
                p1=self._tournament(population, get_key)
                p2=self._tournament(population, get_key)
                child=self._sbx_crossover(p1,p2)
                child=self._poly_mutate(child)
                offspring.append(child)
            population=offspring

            if gen%10==0 and vi:
                best_idx=mfronts[0][0] if mfronts and mfronts[0] else vi[0]
                bo=objs[best_idx]
                LOG.info(f"  NSGA-II Gen {gen:3d} | valid={len(vi)}/{pop_size} | "
                         f"power={bo[0]:.2f} MW | wake={bo[1]:.3f} | "
                         f"load={bo[2]:.4f} | cost={bo[3]/1e6:.2f}M")
                nsga_log.append({'gen':gen,'valid':len(vi),'power_mw':round(bo[0],4),
                                 'wake_ratio':round(bo[1],5),'load_index':round(bo[2],5),
                                 'cost_m':round(bo[3]/1e6,4)})

        save_json({'nsga2_progress': nsga_log}, "nsga2_convergence.json")
        if valid:
            res=self._eval_objectives(best_layout)
            self.final_power=float(res[0]); self.final_wake_ratio=float(res[1])
            self.final_load_index=float(res[2]); self.cable_cost=float(res[3])
        return best_layout

    def _eval_objectives(self, layout):
        n=self.num_turbines; R=self.R; k=EnvironmentConfig.WAKE_DECAY_CONSTANT
        TI_0=EnvironmentConfig.TI_AMBIENT
        k_eff=k*(1.0+0.5*(TI_0+0.1*math.sqrt(0.8)))
        aep=0.0; no_wake=0.0

        all_ti_std=[]  # [v3-2] collect TI std per scenario
        for scen in self.scenarios:
            ws=scen['wind_speed']; wd=scen['wind_direction']; w=scen['weight']
            _, u_eff_s, ti_s = self.cfd_model.compute_power_with_ti(
                layout[:,0], layout[:,1], wind_speed=ws, wind_dir_deg=wd)
            hc=(EnvironmentConfig.Z_HUB/EnvironmentConfig.Z_REF)**EnvironmentConfig.ABL_ALPHA
            p_now=sum(min(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2
                          *(u_eff_s[i]*hc)**3*EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR/1e6,
                          EnvironmentConfig.RATED_POWER) for i in range(n))
            p_no_wk=min(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2
                        *(ws*hc)**3*EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR/1e6,
                        EnvironmentConfig.RATED_POWER)*n
            aep    +=p_now  *w*LifecycleConfig.OPERATING_HOURS
            no_wake+=p_no_wk*w*LifecycleConfig.OPERATING_HOURS
            all_ti_std.append(float(np.std(ti_s)))  # TI variability

        aep_mw=aep/LifecycleConfig.OPERATING_HOURS
        no_wake_mw=no_wake/LifecycleConfig.OPERATING_HOURS
        wake_ratio=max(0.0,(no_wake_mw-aep_mw)/(no_wake_mw+1e-6))
        # [v3-2] Load index = mean Frandsen TI standard deviation across scenarios
        load_idx=float(np.mean(all_ti_std)) if all_ti_std else 0.0

        # Cable cost proxy (MST heuristic)
        G=nx.Graph(); sub_pos=np.array(EnvironmentConfig.GRID_SIZE)/2.0
        full_pos=np.vstack([layout, sub_pos.reshape(1,2)])
        for i in range(n):
            for j in range(i+1, n+1):
                d=np.linalg.norm(full_pos[i]-full_pos[j])
                if d<=EnvironmentConfig.MAX_CABLE_LENGTH: G.add_edge(i,j,weight=d)
        try:
            mst=nx.minimum_spanning_tree(G)
            cable_m=sum(d['weight'] for _,_,d in mst.edges(data=True))
            cable_cost=(cable_m/1000.0)*LifecycleConfig.CABLE_COST_PER_KM
        except Exception: cable_cost=1e9

        md=EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR*EnvironmentConfig.ROTOR_DIAMETER
        feasible=self._is_layout_feasible(layout)
        return (aep_mw, wake_ratio, load_idx, cable_cost, feasible)

    def _fast_non_dom_sort(self, objs):
        n=len(objs); fronts=[[]]; Sp=[[] for _ in range(n)]; np_cnt=[0]*n
        for p in range(n):
            for q in range(n):
                if p==q: continue
                if self._dominates(objs[p],objs[q]): Sp[p].append(q)
                elif self._dominates(objs[q],objs[p]): np_cnt[p]+=1
            if np_cnt[p]==0: fronts[0].append(p)
        i=0
        while fronts[i]:
            nxt=[]
            for p in fronts[i]:
                for q in Sp[p]:
                    np_cnt[q]-=1
                    if np_cnt[q]==0: nxt.append(q)
            i+=1; fronts.append(nxt)
        return [f for f in fronts if f]

    def _crowding_distance(self, objs, fronts):
        """Compute crowding distance for diversity preservation."""
        n=len(objs); cd=[0.0]*n; nobj=len(objs[0])
        for front in fronts:
            if len(front)<=2:
                for idx in front: cd[idx]=float('inf'); continue
            for m in range(nobj):
                front_s=sorted(front, key=lambda i: objs[i][m])
                cd[front_s[0]]=cd[front_s[-1]]=float('inf')
                f_min=objs[front_s[0]][m]; f_max=objs[front_s[-1]][m]
                rng=f_max-f_min+1e-8
                for k in range(1,len(front_s)-1):
                    cd[front_s[k]]+=(objs[front_s[k+1]][m]-objs[front_s[k-1]][m])/rng
        return cd

    def _dominates(self, a, b):
        better=False
        for k in range(len(a)):
            if k==0:
                if a[k]<b[k]: return False
                if a[k]>b[k]: better=True
            else:
                if a[k]>b[k]: return False
                if a[k]<b[k]: better=True
        return better

    def _tournament(self, population, get_key, k=2):
        cands=random.sample(range(len(population)),k)
        cands.sort(key=get_key)
        return population[cands[0]]

    def _sbx_crossover(self, p1, p2, eta=15.0):
        child=np.empty_like(p1)
        for i in range(p1.shape[0]):
            for d in range(2):
                u=np.random.rand()
                beta=(2*u)**(1/(eta+1)) if u<=0.5 else (1/(2*(1-u)))**(1/(eta+1))
                child[i,d]=0.5*((1+beta)*p1[i,d]+(1-beta)*p2[i,d])
        return self._project_feasible(child)

    def _poly_mutate(self, layout, eta=20.0, p_m=None):
        if p_m is None: p_m=1.0/layout.size
        W,H=EnvironmentConfig.GRID_SIZE; pad=ConstraintConfig.BOUNDARY_PADDING
        child=layout.copy()
        for i in range(child.shape[0]):
            for d in range(2):
                if np.random.rand()<p_m:
                    u=np.random.rand()
                    delta=(2*u)**(1/(eta+1))-1 if u<0.5 else 1-(2*(1-u))**(1/(eta+1))
                    lo=pad; hi=(W-pad) if d==0 else (H-pad)
                    child[i,d]=np.clip(child[i,d]+delta*(hi-lo)*0.1, lo, hi)
        return self._project_feasible(child)

    def _layout_violation_score(self, layout):
        W,H=EnvironmentConfig.GRID_SIZE; pad=ConstraintConfig.BOUNDARY_PADDING
        md=EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR*EnvironmentConfig.ROTOR_DIAMETER
        obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN
        score=0.0

        arr=np.asarray(layout, dtype=float)
        if arr.shape!=(self.num_turbines,2) or not np.all(np.isfinite(arr)):
            return 1e12

        score+=float(np.sum(np.maximum(pad-arr[:,0], 0.0)))
        score+=float(np.sum(np.maximum(arr[:,0]-(W-pad), 0.0)))
        score+=float(np.sum(np.maximum(pad-arr[:,1], 0.0)))
        score+=float(np.sum(np.maximum(arr[:,1]-(H-pad), 0.0)))

        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                d=np.linalg.norm(arr[i]-arr[j])
                score+=max(0.0, md-d)

        for p in arr:
            for zone in EnvironmentConfig.FORBIDDEN_ZONES:
                center=np.asarray(zone['center'], dtype=float)
                limit=float(zone['radius'])+obstacle_margin
                score+=max(0.0, limit-np.linalg.norm(p-center))
        return score


    def _is_layout_feasible(self, layout, tol=1e-6):
        return self._layout_violation_score(layout)<=tol


    def _project_feasible(self, layout, iters=None):
        W,H=EnvironmentConfig.GRID_SIZE; pad=ConstraintConfig.BOUNDARY_PADDING
        md=EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR*EnvironmentConfig.ROTOR_DIAMETER
        obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN
        if iters is None:
            iters=EnvironmentConfig.LAYOUT_REPAIR_ITERS

        layout=np.asarray(layout, dtype=float).copy()
        layout[:,0]=np.clip(layout[:,0],pad,W-pad)
        layout[:,1]=np.clip(layout[:,1],pad,H-pad)

        best=layout.copy()
        best_score=self._layout_violation_score(best)

        for _ in range(iters):
            moved=False
            for i in range(len(layout)):
                for j in range(i+1,len(layout)):
                    diff=layout[i]-layout[j]; d=np.linalg.norm(diff)
                    if d<1e-9:
                        angle=np.random.rand()*2.0*math.pi
                        diff=np.array([math.cos(angle), math.sin(angle)])
                        d=1.0
                    if d<md:
                        force=(diff/d)*((md-d)/2.0*1.10)
                        layout[i]+=force; layout[j]-=force; moved=True

            for i in range(len(layout)):
                for zone in EnvironmentConfig.FORBIDDEN_ZONES:
                    center=np.asarray(zone['center'], dtype=float)
                    limit=float(zone['radius'])+obstacle_margin
                    diff=layout[i]-center; d=np.linalg.norm(diff)
                    if d<1e-9:
                        angle=np.random.rand()*2.0*math.pi
                        diff=np.array([math.cos(angle), math.sin(angle)])
                        d=1.0
                    if d<limit:
                        layout[i]=center+(diff/d)*(limit+5.0)
                        moved=True

            layout[:,0]=np.clip(layout[:,0],pad,W-pad)
            layout[:,1]=np.clip(layout[:,1],pad,H-pad)

            score=self._layout_violation_score(layout)
            if score<best_score:
                best=layout.copy(); best_score=score
            if score<=1e-6:
                return layout
            if not moved:
                jitter=np.random.normal(0.0, 0.15*md, size=layout.shape)
                layout+=jitter
                layout[:,0]=np.clip(layout[:,0],pad,W-pad)
                layout[:,1]=np.clip(layout[:,1],pad,H-pad)

        return best

    def _rand_feasible(self):
        W,H=EnvironmentConfig.GRID_SIZE; pad=ConstraintConfig.BOUNDARY_PADDING
        best=None; best_score=float('inf')
        for _ in range(EnvironmentConfig.LAYOUT_REPAIR_ATTEMPTS):
            layout=np.column_stack([
                np.random.uniform(pad, W-pad, self.num_turbines),
                np.random.uniform(pad, H-pad, self.num_turbines),
            ])
            repaired=self._project_feasible(layout)
            score=self._layout_violation_score(repaired)
            if score<best_score:
                best=repaired.copy(); best_score=score
            if score<=1e-6:
                return repaired
        LOG.warning(f"  Layout sampler returned best repair with violation={best_score:.3f}")
        return best if best is not None else self._project_feasible(
            np.random.rand(self.num_turbines,2)*self.grid_size)


# ============================================================================
# SECTION 9: NEURAL NETWORK MODELS
# ============================================================================

class GNNEncoder(nn.Module):
    """
    Two-layer GATv2 graph encoder  [Innovation I3].
    Node features : 18-dimensional [v7: +yaw angle, +yaw derating]
    Edge features : 3-dimensional [dist_norm, res_norm, is_virtual]
    """
    def __init__(self, node_feature_dim=GNNConfig.NODE_FEATURE_DIM,
                 hidden_dim=GNNConfig.HIDDEN_DIM, output_dim=GNNConfig.OUTPUT_DIM):
        super().__init__()
        self.node_norm = nn.LayerNorm(node_feature_dim)
        self.node_emb  = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_emb  = nn.Linear(GNNConfig.EDGE_FEATURE_DIM, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=GNNConfig.HEADS,
                                edge_dim=hidden_dim, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=GNNConfig.HEADS,
                                edge_dim=hidden_dim, concat=False)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout_p   = GNNConfig.DROPOUT

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_norm(x)
        x = F.relu(self.node_emb(x))
        n_edges = edge_index.shape[1]
        if (edge_attr is not None and edge_attr.shape[0]==n_edges
                and edge_attr.shape[1]==GNNConfig.EDGE_FEATURE_DIM and n_edges>0):
            e = self.edge_emb(edge_attr)
        else:
            e = torch.zeros(n_edges, x.size(1), device=x.device)
        x1=self.conv1(x, edge_index, edge_attr=e); x=F.relu(x+x1)
        x=F.dropout(x, p=self.dropout_p, training=self.training)
        x2=self.conv2(x, edge_index, edge_attr=e); x=F.relu(x+x2)
        if batch is None:
            batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g=global_add_pool(x, batch)
        return self.output_proj(g)


class StableDistributionalCritic(nn.Module):
    """IQN-style distributional critic, 51 quantiles """
    def __init__(self, input_dim, n_quantiles=GNNConfig.NUM_QUANTILES):
        super().__init__()
        self.n_quantiles=n_quantiles
        def _mlp():
            return nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim,256),
                nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256,256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, n_quantiles))
        self.online_net=_mlp(); self.target_net=_mlp()
        self.target_net.load_state_dict(self.online_net.state_dict())
        for p in self.target_net.parameters(): p.requires_grad=False
        self.register_buffer('stage_stats', torch.zeros(4,2))
        self.current_stage=0; self.adapt_momentum=0.95
        self.register_buffer('tau', torch.linspace(0.0,1.0,n_quantiles+1)[1:].view(1,-1))

    def forward(self, x, use_target=False, update_stats=False, returns_batch=None, stage=0):
        net=self.target_net if use_target else self.online_net; q=net(x)
        if update_stats and returns_batch is not None and self.training:
            with torch.no_grad():
                if stage!=self.current_stage:
                    self.current_stage=stage; self.adapt_momentum=0.80
                else:
                    self.adapt_momentum=min(0.99,self.adapt_momentum+0.001)
                m=returns_batch.mean(); s=returns_batch.std()+1e-6
                self.stage_stats[stage,0]=(self.adapt_momentum*self.stage_stats[stage,0]
                                            +(1-self.adapt_momentum)*m)
                self.stage_stats[stage,1]=(self.adapt_momentum*self.stage_stats[stage,1]
                                            +(1-self.adapt_momentum)*s)
        centre=self.stage_stats[stage,0]; scale=self.stage_stats[stage,1].clamp(min=10.0)
        return q*scale+centre

    def get_value(self, x, use_target=False, stage=0):
        q=self.forward(x, use_target=use_target, stage=stage)
        sq,_=torch.sort(q,dim=1); trim=max(1,int(self.n_quantiles*0.1))
        return sq[:,trim:-trim].mean(dim=1,keepdim=True)

    def soft_update_target(self):
        tau=PPOConfig.POLYAK_TAU
        for po,pt in zip(self.online_net.parameters(), self.target_net.parameters()):
            pt.data.copy_(tau*po.data+(1-tau)*pt.data)


class GNNActorCritic(nn.Module):
    """Joint actor-critic sharing the GATv2 encoder.

    The actor is hybrid: a categorical head selects the cable-topology edit,
    while a Gaussian continuous head commands turbine yaw angles.
    """
    def __init__(self, gnn_encoder, action_dim):
        super().__init__()
        self.gnn   = gnn_encoder
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(GNNConfig.OUTPUT_DIM,128), nn.Tanh(),
            nn.Linear(128,64), nn.Tanh(),
            nn.Linear(64, action_dim))
        self.yaw_head = nn.Sequential(
            nn.Linear(GNNConfig.OUTPUT_DIM,128), nn.Tanh(),
            nn.Linear(128,64), nn.Tanh(),
            nn.Linear(64, EnvironmentConfig.NUM_TURBINES))
        self.yaw_log_std = nn.Parameter(torch.full(
            (EnvironmentConfig.NUM_TURBINES,),
            math.log(EnvironmentConfig.YAW_MAX_DEG*0.25)))
        self.critic= StableDistributionalCritic(GNNConfig.OUTPUT_DIM)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None: m.bias.data.zero_()

    def get_value(self, x, ei, ea, batch=None, use_target=False, stage=0):
        emb=self.gnn(x, ei, ea, batch)
        return self.critic.get_value(emb, use_target=use_target, stage=stage)

    def get_action_and_value(self, x, ei, ea, batch=None, action=None, yaw_action=None,
                              valid_mask=None, use_target=False, stage=0):
        emb   =self.gnn(x, ei, ea, batch)
        value =self.critic.get_value(emb, use_target=use_target, stage=stage)
        logits=self.actor(emb)
        if valid_mask is not None:
            if valid_mask.dim()==1: valid_mask=valid_mask.unsqueeze(0)
            logits=logits.masked_fill(~valid_mask, -1e8)
        dist=Categorical(logits=logits)
        if action is None: action=dist.sample()

        yaw_max = EnvironmentConfig.YAW_MAX_DEG
        yaw_mean = torch.tanh(self.yaw_head(emb))*yaw_max
        yaw_std = self.yaw_log_std.exp().clamp(min=0.5, max=yaw_max).view(1,-1).expand_as(yaw_mean)
        yaw_dist = Normal(yaw_mean, yaw_std)
        if yaw_action is None:
            yaw_action = yaw_dist.sample()
            yaw_action = torch.clamp(yaw_action, -yaw_max, yaw_max)
        else:
            if yaw_action.dim()==1: yaw_action=yaw_action.unsqueeze(0)
            yaw_action = torch.clamp(yaw_action.to(yaw_mean.device), -yaw_max, yaw_max)

        topo_logprob = dist.log_prob(action)
        yaw_logprob  = yaw_dist.log_prob(yaw_action).sum(dim=-1)/EnvironmentConfig.NUM_TURBINES
        topo_entropy = dist.entropy()
        yaw_entropy  = yaw_dist.entropy().sum(dim=-1)/EnvironmentConfig.NUM_TURBINES
        return action, yaw_action, topo_logprob+yaw_logprob, topo_entropy+0.10*yaw_entropy, value


# ============================================================================
# SECTION 10: ACTION MAPPER
# ============================================================================

class ActionMapper:
    """
    Geometrically constrained action space.
    Each connection (u,v) → two actions: idx*2+0=remove, idx*2+1=add.
    """
    def __init__(self, n_turbines, n_substations, node_positions):
        self.num_turbines=n_turbines; self.num_nodes=n_turbines+n_substations
        self.node_positions=node_positions
        self.connections=self._generate_connections()
        self.action_dim=len(self.connections)*2
        self.conn_lines=[LineString([node_positions[u],node_positions[v]])
                         for u,v in self.connections]
        self.forbidden_polys=[Point(z['center']).buffer(z['radius'])
                               for z in EnvironmentConfig.FORBIDDEN_ZONES]
        LOG.info(f"  ActionMapper | connections={len(self.connections)} action_dim={self.action_dim}")

    def _generate_connections(self):
        conns=[]
        for i in range(self.num_nodes):
            for j in range(i+1,self.num_nodes):
                d=np.linalg.norm(self.node_positions[i]-self.node_positions[j])
                if d<=EnvironmentConfig.MAX_CABLE_LENGTH: conns.append((i,j))
        return sorted(conns)

    def map_action(self, idx):
        ci=idx//2; at=idx%2; u,v=self.connections[ci]; return (u,v,at)

    def get_action_space_size(self): return self.action_dim

    def get_valid_mask(self, current_edges):
        mask=torch.zeros(self.action_dim, dtype=torch.bool)
        valid=self.get_valid_indices(current_edges)
        if valid: mask[valid]=True
        else: mask[0]=True
        return mask

    def get_valid_indices(self, current_edges):
        edge_set={tuple(sorted((u,v))) for u,v in current_edges}
        valid=[]
        for idx in range(self.action_dim):
            ci=idx//2; is_add=(idx%2==1); u,v=self.connections[ci]
            exists=tuple(sorted((u,v))) in edge_set
            if is_add:
                if exists: continue
                line=self.conn_lines[ci]
                if any(line.intersects(p) and not line.touches(p) for p in self.forbidden_polys):
                    continue
                valid.append(idx)
            else:
                if exists: valid.append(idx)
        return valid


# ============================================================================
# SECTION 11: TOPOLOGY DATA GENERATOR  [Innovation I3]
# ============================================================================

class TopologyDataGenerator:
    """Generates 800 labelled samples for GNN multi-task pre-training."""
    def __init__(self, env):
        self.env=env

    def generate(self, n_samples, fixed_layout, mapper):
        LOG.info(f"  Generating {n_samples} topology samples ...")
        samples=[]
        for i in range(n_samples):
            if i % 100 == 0:
                LOG.info('[BASE SAMPLES] %d/%d', i, n_samples)
            self.env.reset(use_mst_init=False)
            self.env.node_positions[:self.env.num_turbines]=fixed_layout
            self.env.turbine_positions=fixed_layout.copy()
            self.env._precompute_wake()
            topo=(self._mst_topo() if i%3==0 else
                  self._rand_topo() if i%3==1 else self._radial_topo())
            for u in topo:
                for v in topo[u]:
                    if u<v: self.env._add_edge(u,v)
            self.env._update_node_features()
            ohmic    =self.env._ohmic_loss()
            length_km=self.env._total_cable_length()/1000.0
            n_comp   =float(nx.number_connected_components(self.env._build_nx_graph()))
            nf,ei,ea =self.env._get_state()
            rows,cols=np.where(self.env.adjacency_matrix>0)
            curr_edges=[(r,c) for r,c in zip(rows,cols) if r<c]
            mst_action=self._mst_target_action(curr_edges, nf, mapper)
            samples.append({'state':(nf,ei,ea),
                            'targets':{'ohmic':float(ohmic),'length':float(length_km),
                                       'components':float(n_comp),'action_idx':mst_action}})
        LOG.info(f"  Generated {len(samples)} labelled samples")
        return samples

    def _mst_topo(self):
        G=nx.Graph()
        for i in range(self.env.num_nodes):
            for j in range(i+1,self.env.num_nodes):
                d=np.linalg.norm(self.env.node_positions[i]-self.env.node_positions[j])*random.uniform(0.5,1.5)
                G.add_edge(i,j,weight=d)
        mst=nx.minimum_spanning_tree(G); topo=defaultdict(list)
        for u,v in mst.edges(): topo[u].append(v); topo[v].append(u)
        return topo

    def _rand_topo(self, p=0.25):
        topo=self._mst_topo()
        for i in range(self.env.num_nodes):
            for j in range(i+1,self.env.num_nodes):
                if j not in topo[i] and random.random()<p:
                    d=np.linalg.norm(self.env.node_positions[i]-self.env.node_positions[j])
                    if d<=EnvironmentConfig.MAX_CABLE_LENGTH: topo[i].append(j); topo[j].append(i)
        return topo

    def _radial_topo(self):
        topo=defaultdict(list); sub=self.env.num_nodes-1
        for i in range(self.env.num_turbines):
            topo[i].append(sub); topo[sub].append(i)
            dists=sorted((np.linalg.norm(self.env.node_positions[i]-self.env.node_positions[j]),j)
                         for j in range(self.env.num_turbines) if j!=i)
            if dists:
                nn=dists[0][1]
                if nn not in topo[i]: topo[i].append(nn); topo[nn].append(i)
        return topo

    def _mst_target_action(self, curr_edges, nf, mapper):
        G=nx.Graph()
        W,H=EnvironmentConfig.GRID_SIZE
        pos=nf[:,:2].numpy()*np.array([W,H])
        for i in range(self.env.num_nodes):
            for j in range(i+1,self.env.num_nodes): G.add_edge(i,j,weight=np.linalg.norm(pos[i]-pos[j]))
        mst=nx.minimum_spanning_tree(G); curr=set(curr_edges)
        for u,v in mst.edges():
            u,v=sorted((u,v))
            if (u,v) not in curr:
                for ci,conn in enumerate(mapper.connections):
                    if conn==(u,v): return ci*2+1
        return 0


# ============================================================================
# SECTION 12: MULTI-TASK GNN PRE-TRAINER  [Innovation I3]
# ============================================================================

class MultiTaskPretrainer:
    """
    Multi-task supervised pre-training of GATv2 encoder.
    Tasks: ohmic_loss, cable_length, n_components, action_logits.
    """
    def __init__(self, gnn, device, action_dim):
        self.gnn=gnn; self.device=device
        self.heads=nn.ModuleDict({
            'ohmic'       : nn.Linear(GNNConfig.OUTPUT_DIM,1),
            'length'      : nn.Linear(GNNConfig.OUTPUT_DIM,1),
            'components'  : nn.Linear(GNNConfig.OUTPUT_DIM,1),
            'action_logits': nn.Linear(GNNConfig.OUTPUT_DIM, action_dim),
        }).to(device)
        self.optimizer=torch.optim.Adam(
            list(gnn.parameters())+list(self.heads.parameters()), lr=GNNConfig.PRETRAIN_LR)
        self.lr_sched=ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)
        self.best_val=float('inf'); self.patience_cnt=0; self.patience=25
        self.best_state=None; self.train_losses=[]; self.val_losses=[]

    def train(self, samples, epochs, mapper, env):
        log_section("Phase 2 – GATv2 Multi-Task GNN Pre-training  [I3]")
        LOG.info(f"  Samples={len(samples)}  Epochs={epochs}  Patience={self.patience}  LR={GNNConfig.PRETRAIN_LR:.0e}")
        random.shuffle(samples)
        split=int(len(samples)*0.8)
        t_loader=self._make_loader(samples[:split])
        v_loader=self._make_loader(samples[split:])

        for ep in range(epochs):
            self.gnn.train(); self.heads.train()
            t_loss=self._epoch(t_loader, train=True)
            self.gnn.eval(); self.heads.eval()
            with torch.no_grad(): v_loss=self._epoch(v_loader, train=False)
            self.train_losses.append(t_loss); self.val_losses.append(v_loss)
            self.lr_sched.step(v_loss)
            if v_loss<self.best_val-1e-6:
                self.best_val=v_loss; self.patience_cnt=0
                self.best_state={'gnn':{k:v.clone() for k,v in self.gnn.state_dict().items()},
                                 'heads':{k:v.clone() for k,v in self.heads.state_dict().items()}}
            else: self.patience_cnt+=1
            if ep%20==0 or self.patience_cnt==self.patience:
                LOG.info(f"  Pretrain ep={ep:4d} | train={t_loss:.5f} | val={v_loss:.5f} | "
                         f"best={self.best_val:.5f} | patience={self.patience_cnt}/{self.patience}")
            if self.patience_cnt>=self.patience:
                LOG.info(f"  Early stopping at ep={ep}. Restoring best.")
                if self.best_state:
                    self.gnn.load_state_dict(self.best_state['gnn'])
                    self.heads.load_state_dict(self.best_state['heads'])
                break

        val_stats=self._validate(v_loader)
        LOG.info(f"  Pretraining complete | action_acc={val_stats['action_acc']:.4f} | "
                 f"ohmic_mae={val_stats['ohmic_mae']:.5f}")
        save_json({'train_losses':self.train_losses,'val_losses':self.val_losses,
                   'best_val':self.best_val,'action_acc':val_stats['action_acc'],
                   'ohmic_mae':val_stats['ohmic_mae']}, "pretrain_curves.json")
        return val_stats

    def _epoch(self, loader, train):
        total=0.0
        for batch in loader:
            batch=batch.to(self.device)
            emb=self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            p_ohm=self.heads['ohmic'](emb).squeeze(-1)
            p_len=self.heads['length'](emb).squeeze(-1)
            p_comp=self.heads['components'](emb).squeeze(-1)
            p_act=self.heads['action_logits'](emb)
            l1=F.mse_loss(p_ohm,batch.y_ohmic); l2=F.mse_loss(p_len,batch.y_len)
            l3=F.mse_loss(p_comp,batch.y_comp); l4=F.cross_entropy(p_act,batch.y_action)
            loss=l1+l2+l3+0.5*l4
            if train:
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.gnn.parameters())+list(self.heads.parameters()),1.0)
                self.optimizer.step()
            total+=loss.item()
        return total/max(len(loader),1)

    def _validate(self, loader):
        self.gnn.eval(); self.heads.eval()
        all_ap,all_at,all_op,all_ot=[],[],[],[]
        with torch.no_grad():
            for batch in loader:
                batch=batch.to(self.device)
                emb=self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                p_act=self.heads['action_logits'](emb); p_ohm=self.heads['ohmic'](emb).squeeze(-1)
                all_ap.append(p_act.argmax(1).cpu()); all_at.append(batch.y_action.cpu())
                all_op.append(p_ohm.cpu()); all_ot.append(batch.y_ohmic.cpu())
        acc=(torch.cat(all_ap)==torch.cat(all_at)).float().mean().item()
        mae=(torch.cat(all_op)-torch.cat(all_ot)).abs().mean().item()
        return {'action_acc':acc,'ohmic_mae':mae}

    def _make_loader(self, samples):
        data_list=[]
        for s in samples:
            nf,ei,ea=s['state']
            if ea is None or ea.numel()==0:
                ea=torch.zeros((0,GNNConfig.EDGE_FEATURE_DIM),dtype=torch.float32)
            d=Data(x=nf, edge_index=ei, edge_attr=ea,
                   y_ohmic=torch.tensor([s['targets']['ohmic']],dtype=torch.float32),
                   y_len  =torch.tensor([s['targets']['length']],dtype=torch.float32),
                   y_comp =torch.tensor([s['targets']['components']],dtype=torch.float32),
                   y_action=torch.tensor([s['targets']['action_idx']],dtype=torch.long))
            data_list.append(d)
        return DataLoader(data_list, batch_size=GNNConfig.PRETRAIN_BATCH_SIZE, shuffle=True)


# ============================================================================
# SECTION 13: CURRICULUM SCHEDULER
# ============================================================================

class AdaptiveCurriculumScheduler:
    """Four-stage curriculum with relaxed advancement criteria."""
    def __init__(self, env):
        self.env=env; self.stages=CurriculumConfig.STAGES; self.cur=0
        self.entry_ep=0; self.perf_buf=deque(maxlen=50); self.success_buf=deque(maxlen=50)
        self.cost_buf=deque(maxlen=50); self.consec=0; self.stage_log=[]

    def step(self, episode, result):
        self.perf_buf.append(result); self.success_buf.append(result['success'])
        if result['cost']>0: self.cost_buf.append(result['cost'])
        if result['success']: self.consec+=1
        else: self.consec=0
        stage=self.stages[self.cur]
        self.env.cost_weight=stage['cost_weight_target']
        self.env.max_steps=EnvironmentConfig.MAX_STEPS+stage['max_steps_offset']
        self.env.reward_mode=stage['reward_mode']
        self.env.max_allowed_disconnections=stage['max_disconnections']
        in_stage=episode-self.entry_ep
        if (in_stage>=stage['min_episodes'] and self.cur<len(self.stages)-1
                and self._check_advancement(stage)):
            prev=self.cur; self.cur+=1; self.entry_ep=episode
            self.perf_buf.clear(); self.success_buf.clear(); self.cost_buf.clear(); self.consec=0
            LOG.info(f"[CURRICULUM] Stage {prev}({stage['name']}) → {self.cur}"
                     f"({self.stages[self.cur]['name']}) at ep={episode}")
            self.stage_log.append({'episode':episode,'from':prev,'to':self.cur})
        return stage, self.cur

    def _check_advancement(self, stage):
        crit=stage['advancement_criteria']
        if crit is None or len(self.perf_buf)<30: return False
        sr=float(np.mean(list(self.success_buf)))
        costs=list(self.cost_buf); cv=999.0
        if len(costs)>10:
            med=float(np.median(costs)); mad=float(np.median(np.abs(np.array(costs)-med)))
            cv=mad/(med+1e-8)
        avg_iso=float(np.mean([p['isolated'] for p in list(self.perf_buf)[-30:]]))
        return (sr>=crit['min_success_rate'] and cv<=crit.get('max_cost_cv',0.5)
                and avg_iso<=crit['max_isolated'] and self.consec>=crit.get('min_consecutive_success',9999))

    def get_stage_transitions(self): return self.stage_log


# ============================================================================
# SECTION 14: PPO AGENT
# ============================================================================

class PPOAgent:
    """PPO with distributional critic and cosine LR annealing [Innovation I4, v3-5]."""
    def __init__(self, actor_critic, device, total_episodes=TrainingConfig.NUM_EPISODES):
        self.net=actor_critic.to(device); self.device=device
        self.opt=optim.Adam(actor_critic.parameters(), lr=PPOConfig.LR_INIT, eps=1e-5)
        self.total_episodes=total_episodes
        self.last_pg_loss=0.0; self.last_v_loss=0.0
        self.last_entropy=0.0; self.last_approx_kl=0.0
        self._last_stage=0; self._stage_steps=0
        # [v6-2] Entropy refresh tracking
        self._low_entropy_streak=0
        self._last_refresh_ep=0

    def update_lr(self, episode):
        """[v3-5] Cosine annealing from LR_INIT to LR_FINAL."""
        t=episode/self.total_episodes
        lr=PPOConfig.LR_FINAL+(PPOConfig.LR_INIT-PPOConfig.LR_FINAL)*0.5*(1+math.cos(math.pi*t))
        for pg in self.opt.param_groups: pg['lr']=lr

    def check_and_refresh_entropy(self, episode, stage):
        """[v6-2] Entropy refresh: if H collapses below 0.12 in stage 3 for ≥40
        consecutive episodes, apply controlled noise to actor parameters to
        restore diversity. Rate-limited: at most once every 200 episodes.
        """
        if stage < 3: self._low_entropy_streak = 0; return
        if self.last_entropy < 0.12:
            self._low_entropy_streak += 1
        else:
            self._low_entropy_streak = max(0, self._low_entropy_streak - 1)

        cooldown_ok = (episode - self._last_refresh_ep) >= 200
        if self._low_entropy_streak >= 40 and cooldown_ok:
            with torch.no_grad():
                for name, param in self.net.actor.named_parameters():
                    sigma = 0.03 if 'weight' in name else 0.01
                    param.data.add_(torch.randn_like(param) * sigma)
            self._low_entropy_streak = 0
            self._last_refresh_ep = episode
            LOG.info(f"  [v6-2] Entropy refresh applied at ep={episode} "
                     f"(H={self.last_entropy:.3f} < 0.12 for 40+ eps)")

    def _adaptive_ent_coef(self, stage):
        """ PID-style entropy control towards per-stage target_entropy.
         Raised minimum entropy coefficient floor from 5e-4 to 2e-3 to
        prevent over-determinism in stage 3 (entropy near 0.15-0.30) which
        caused occasional connectivity failures (iso>0 at ~2–8% of episodes).
        """
        stage_cfg = CurriculumConfig.STAGES[min(stage, len(CurriculumConfig.STAGES)-1)]
        target_h  = stage_cfg['target_entropy']
        h_err     = self.last_entropy - target_h  # positive = too high

        if   h_err >  2.0: mult = 0.05   # massively above target → suppress hard
        elif h_err >  1.0: mult = 0.15
        elif h_err >  0.5: mult = 0.40
        elif h_err > -0.1: mult = 1.00   # near target
        elif h_err > -0.5: mult = 2.00   # slightly too low → boost
        else:              mult = 3.00   # too deterministic → explore more

        # Also modulate on approx KL
        if self.last_approx_kl > 0.05: mult *= 0.7
        elif self.last_approx_kl < 0.005: mult *= 1.3

        # Stage-decaying base coefficient
        base = [0.030, 0.020, 0.010, 0.005][min(stage, 3)]
        # [v5-5] Raised floor: 2e-3 (was 5e-4) to maintain minimum exploration
        coef = max(2e-3, base * mult)
        return coef

    @torch.no_grad()
    def get_action(self, state, valid_mask=None, use_target=False, stage=0):
        self.net.eval()
        nf,ei,ea=state
        nf=nf.to(self.device); ei=ei.to(self.device)
        ea=ea.to(self.device) if ea is not None else None
        vm=valid_mask.to(self.device) if valid_mask is not None else None
        act,yaw,lp,ent,val=self.net.get_action_and_value(
            nf,ei,ea,batch=None,valid_mask=vm,use_target=use_target,stage=stage)
        self.last_entropy=ent.mean().item() if ent is not None else 0.0
        return act.item(), yaw.squeeze(0).detach().cpu().numpy(), lp.item(), val.item()

    def update(self, rollouts, stage=0):
        self.net.train()
        if stage>self._last_stage: self._last_stage=stage; self._stage_steps=0
        self._stage_steps+=len(rollouts['obs'])
        ent_coef=self._adaptive_ent_coef(stage)
        returns_t=torch.tensor(rollouts['returns'],dtype=torch.float32)
        adv_t=torch.tensor(rollouts['advantages'],dtype=torch.float32)
        adv_t=(adv_t-adv_t.mean())/(adv_t.std()+1e-8)

        data_list=[]
        for i in range(len(rollouts['obs'])):
            nf,ei,ea=rollouts['obs'][i]
            if ea is None or ea.numel()==0:
                ea=torch.zeros((0,GNNConfig.EDGE_FEATURE_DIM),dtype=torch.float32)
            d=Data(x=nf,edge_index=ei,edge_attr=ea)
            d.action  =torch.tensor(rollouts['actions'][i],dtype=torch.long)
            d.yaw_action=torch.tensor(rollouts['yaw_actions'][i],dtype=torch.float32).view(1,-1)
            d.logprob =torch.tensor(rollouts['logprobs'][i],dtype=torch.float32)
            d.advantage=adv_t[i]; d.returns=returns_t[i]
            vm=rollouts['valid_masks'][i]; d.valid_mask=vm.bool()
            data_list.append(d)

        loader=DataLoader(data_list, batch_size=PPOConfig.BATCH_SIZE, shuffle=True)
        for _ in range(PPOConfig.K_EPOCHS):
            for batch in loader:
                batch=batch.to(self.device)
                action_dim=rollouts['valid_masks'][0].shape[0]
                vm_batch=batch.valid_mask.view(-1,action_dim)
                emb=self.net.gnn(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
                online_q=self.net.critic(emb,update_stats=True,returns_batch=batch.returns,
                                         use_target=False,stage=stage)
                tgt=batch.returns.unsqueeze(1).unsqueeze(2); preds=online_q.unsqueeze(2)
                td=tgt-preds; kappa=1.0
                huber=torch.where(td.abs()<=kappa, 0.5*td.pow(2), kappa*(td.abs()-0.5*kappa))
                tau=self.net.critic.tau.to(self.device)
                q_loss=((tau-(td<0).float()).abs()*huber).mean()
                yaw_batch=batch.yaw_action.view(-1, EnvironmentConfig.NUM_TURBINES)
                _,_,new_lp,entropy,_=self.net.get_action_and_value(
                    batch.x,batch.edge_index,batch.edge_attr,batch.batch,
                    action=batch.action,yaw_action=yaw_batch,
                    valid_mask=vm_batch,use_target=False,stage=stage)
                log_ratio=new_lp-batch.logprob; ratio=log_ratio.exp()
                with torch.no_grad():
                    self.last_approx_kl=((ratio-1)-log_ratio).mean().item()
                adv=batch.advantage
                pg=torch.max(-adv*ratio,
                              -adv*ratio.clamp(1-PPOConfig.CLIP_EPSILON,1+PPOConfig.CLIP_EPSILON)).mean()
                loss=pg-ent_coef*entropy.mean()+PPOConfig.VALUE_LOSS_COEF*q_loss
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), PPOConfig.MAX_GRAD_NORM)
                self.opt.step(); self.net.critic.soft_update_target()
                self.last_pg_loss=pg.item(); self.last_v_loss=q_loss.item()
                self.last_entropy=entropy.mean().item()


# ============================================================================
# SECTION 15: PPO TRAINER
# ============================================================================

class PPOTrainer:
    """Episode-based PPO with curriculum scheduling and cosine LR [v3-5]."""
    def __init__(self, env, agent, mapper):
        self.env=env; self.agent=agent; self.mapper=mapper
        self.curriculum=AdaptiveCurriculumScheduler(env)
        self.best_topo=None; self.min_cost=float('inf')
        # [v6-3] Secondary best tracker: cost-only (iso=0, obs=0, any crossings)
        self.best_cost_topo=None; self.min_cost_unrestricted=float('inf')
        self.best_reliable_topo=None; self.min_reliable_score=float('inf')
        self.best_near_reliable_topo=None; self.min_near_reliable_score=float('inf')
        self.reliability_candidates=[]
        self.episode_log=[]

    def _snapshot_topology(self, cost_m, episode, val_check, ci, tag):
        return {'adj_matrix':self.env.adjacency_matrix.copy(),
                'edge_index':self.env.edge_index.clone(),
                'edge_attr' :self.env.edge_attr.clone(),
                'yaw_angles':self.env.yaw_angles.copy(),
                'cost_m':cost_m,'episode':episode,'validation':val_check,
                'reliability':{
                    'c_reliability_m':ci.get('c_reliability',0.0)/1e6,
                    'redundancy_ratio':ci.get('redundancy_ratio',0.0),
                    'max_single_edge_outage_turbines':ci.get('max_single_edge_outage_turbines',0),
                    'expected_unserved_energy_mwh':ci.get('expected_unserved_energy_mwh',0.0),
                    'n_bridge_edges':ci.get('n_bridge_edges',0),
                    'n_cycle_edges':ci.get('n_cycle_edges',0),
                    'crossing_violations':val_check.get('crossing_violations',0),
                },
                'tag':tag}

    def _reliability_selection_score(self, ci, cost_m):
        n1=ci.get('max_single_edge_outage_turbines', self.env.num_turbines)
        ens=ci.get('expected_unserved_energy_mwh', 1e9)
        red=ci.get('redundancy_ratio', 0.0)
        rel_m=ci.get('c_reliability',0.0)/1e6
        cross=ci.get('crossing_violations', 0)
        n1_gap=max(0, n1-ReliabilityConfig.RELIABLE_MAX_N1_OUTAGE_TURBINES)
        ens_gap=max(0.0, ens-ReliabilityConfig.RELIABLE_MAX_ENS_MWH)
        red_gap=max(0.0, ReliabilityConfig.RELIABLE_MIN_REDUNDANCY_RATIO-red)
        cross_gap=max(0, cross-ConstraintConfig.MAX_CROSSING_FOR_BEST)
        return (cost_m + 0.35*rel_m + 0.18*n1 + 0.0015*ens
                + 0.45*cross - 2.0*red
                + 1.75*n1_gap + 0.006*ens_gap + 10.0*red_gap
                + 0.85*cross_gap)

    def _is_reliable_candidate(self, ci):
        return (
            ci.get('max_single_edge_outage_turbines', self.env.num_turbines)
            <= ReliabilityConfig.RELIABLE_MAX_N1_OUTAGE_TURBINES
            and ci.get('redundancy_ratio', 0.0)
            >= ReliabilityConfig.RELIABLE_MIN_REDUNDANCY_RATIO
            and ci.get('expected_unserved_energy_mwh', float('inf'))
            <= ReliabilityConfig.RELIABLE_MAX_ENS_MWH
        )

    def _is_near_reliable_candidate(self, ci):
        return (
            ci.get('max_single_edge_outage_turbines', self.env.num_turbines)
            <= ReliabilityConfig.NEAR_RELIABLE_MAX_N1_OUTAGE_TURBINES
            and ci.get('redundancy_ratio', 0.0)
            >= ReliabilityConfig.NEAR_RELIABLE_MIN_REDUNDANCY_RATIO
            and ci.get('expected_unserved_energy_mwh', float('inf'))
            <= ReliabilityConfig.NEAR_RELIABLE_MAX_ENS_MWH
        )

    def _archive_reliability_candidate(self, ep_cnt, cost_m, ci, val_check, score, gate):
        cand={'episode':ep_cnt,'cost_m':round(cost_m,4),'score':round(score,4),
              'gate':gate,
              'redundancy_ratio':round(ci.get('redundancy_ratio',0.0),4),
              'n1_outage_turbines':ci.get('max_single_edge_outage_turbines',0),
              'unserved_mwh':round(ci.get('expected_unserved_energy_mwh',0.0),4),
              'c_reliability_m':round(ci.get('c_reliability',0.0)/1e6,4),
              'crossing_violations':val_check.get('crossing_violations',0),
              'crossing_compliant':val_check.get('crossing_violations',0)<=ConstraintConfig.MAX_CROSSING_FOR_BEST}
        self.reliability_candidates.append(cand)
        self.reliability_candidates=sorted(
            self.reliability_candidates, key=lambda r: r['score'])[:25]

    def train(self, total_episodes):
        log_section("Phase 3 – Curriculum-Guided PPO Training  [I4]")
        LOG.info(f"  Episodes={total_episodes}  K_epochs={PPOConfig.K_EPOCHS}  "
                 f"Gamma={PPOConfig.GAMMA}  Lambda={PPOConfig.GAE_LAMBDA}")
        LOG.info(f"  [v3-4] Crossing threshold for best-topo: ≤{ConstraintConfig.MAX_CROSSING_FOR_BEST}")
        LOG.info(f"  [v3-5] Cosine LR annealing: {PPOConfig.LR_INIT:.0e} → {PPOConfig.LR_FINAL:.0e}")

        WARMUP=50
        obs_b,act_b,yaw_b,lp_b,rew_b,val_b,mask_b,vm_b=[],[],[],[],[],[],[],[]
        state=self.env.reset(use_mst_init=True)
        ep_rew=0.0; ep_cnt=0; rewards_hist=[]; conn_hist=[]

        def up_steps(ep):
            return 256 if ep<WARMUP else (512 if ep<200 else 1024)

        while ep_cnt<total_episodes:
            if ep_cnt<WARMUP:
                self.env.cost_weight=0.0; self.env.max_allowed_disconnections=15
                self.curriculum.cur=0
            self.agent.update_lr(ep_cnt)  # [v3-5]
            done=False
            for _ in range(up_steps(ep_cnt)):
                vm=self.mapper.get_valid_mask(self.env._get_unique_edges())
                act,yaw,lp,val=self.agent.get_action(state,vm,use_target=False,stage=self.curriculum.cur)
                _,_,_,val_t=self.agent.get_action(state,vm,use_target=True,stage=self.curriculum.cur)
                mapped_act=self.mapper.map_action(act)
                next_state,rew,done,info=self.env.step((mapped_act, yaw))
                obs_b.append(state); act_b.append(act); yaw_b.append(yaw); lp_b.append(lp)
                rew_b.append(rew); val_b.append(val_t); mask_b.append(0 if done else 1)
                vm_b.append(vm); state=next_state; ep_rew+=rew

                if done:
                    ci=self.env.calculate_lifecycle_cost()
                    cm=self.env.calculate_control_metrics()
                    cost_m=ci['total']/1e6
                    iso_cnt=self.env._count_isolated_turbines()
                    val_check=self.env.validate_topology()
                    ci['crossing_violations']=val_check.get('crossing_violations', 0)
                    conn_hist.append(iso_cnt)

                    #  Best topology requires crossings ≤ MAX_CROSSING_FOR_BEST
                    if val_check['is_valid'] and cost_m<self.min_cost:
                        self.min_cost=cost_m
                        self.best_topo=self._snapshot_topology(cost_m, ep_cnt, val_check, ci, 'min_cost_valid')
                        LOG.info(f"[BEST TOPOLOGY] ep={ep_cnt} cost={cost_m:.3f}M | "
                                 f"iso={iso_cnt} | obs={val_check['obstacle_violations']} | "
                                 f"cross={val_check['crossing_violations']}")

                    #  Secondary tracker: best cost ignoring crossing constraint
                    if (iso_cnt==0 and val_check['obstacle_violations']==0
                            and cost_m<self.min_cost_unrestricted):
                        self.min_cost_unrestricted=cost_m
                        self.best_cost_topo=self._snapshot_topology(cost_m, ep_cnt, val_check, ci, 'min_cost_unrestricted')
                        self.best_cost_topo['crossing_violations']=val_check['crossing_violations']
                        LOG.info(f"[BEST COST (unrestricted)] ep={ep_cnt} cost={cost_m:.3f}M | "
                                 f"cross={val_check['crossing_violations']}")

                    rel_candidate_ok = (
                        iso_cnt==0 and val_check['obstacle_violations']==0
                        and self._is_reliable_candidate(ci)
                    )
                    near_candidate_ok = (
                        iso_cnt==0 and val_check['obstacle_violations']==0
                        and self._is_near_reliable_candidate(ci)
                    )
                    if rel_candidate_ok:
                        score=self._reliability_selection_score(ci, cost_m)
                        self._archive_reliability_candidate(
                            ep_cnt, cost_m, ci, val_check, score, 'strict')
                        if score<self.min_reliable_score:
                            self.min_reliable_score=score
                            self.best_reliable_topo=self._snapshot_topology(
                                cost_m, ep_cnt, val_check, ci, 'reliable_compromise')
                            self.best_reliable_topo['selection_score']=score
                            LOG.info(f"[BEST RELIABLE TOPOLOGY] ep={ep_cnt} cost={cost_m:.3f}M | "
                                     f"score={score:.3f} | red={ci.get('redundancy_ratio',0.0):.3f} | "
                                     f"N1={ci.get('max_single_edge_outage_turbines',0)} | "
                                     f"ENS={ci.get('expected_unserved_energy_mwh',0.0):.1f} MWh | "
                                     f"cross={val_check.get('crossing_violations',0)}")
                    elif near_candidate_ok:
                        score=self._reliability_selection_score(ci, cost_m)
                        self._archive_reliability_candidate(
                            ep_cnt, cost_m, ci, val_check, score, 'near')
                        if score<self.min_near_reliable_score:
                            self.min_near_reliable_score=score
                            self.best_near_reliable_topo=self._snapshot_topology(
                                cost_m, ep_cnt, val_check, ci, 'near_reliable_compromise')
                            self.best_near_reliable_topo['selection_score']=score
                            LOG.info(f"[BEST NEAR-RELIABLE TOPOLOGY] ep={ep_cnt} cost={cost_m:.3f}M | "
                                     f"score={score:.3f} | red={ci.get('redundancy_ratio',0.0):.3f} | "
                                     f"N1={ci.get('max_single_edge_outage_turbines',0)} | "
                                     f"ENS={ci.get('expected_unserved_energy_mwh',0.0):.1f} MWh | "
                                     f"cross={val_check.get('crossing_violations',0)}")

                    ep_res={'success':iso_cnt==0,'isolated':iso_cnt,'cost':cost_m}
                    stage,si=self.curriculum.step(ep_cnt,ep_res)
                    # [v6-2] Check and refresh entropy if collapsed
                    self.agent.check_and_refresh_entropy(ep_cnt, si)
                    rewards_hist.append(ep_rew)

                    self.episode_log.append({
                        'episode':ep_cnt,'stage':si,'reward':round(ep_rew,3),
                        'cost_m':round(cost_m,4),'isolated':iso_cnt,
                        'c_inv_m':round(ci['c_inv']/1e6,4),'c_op_m':round(ci['c_op']/1e6,4),
                        'c_wake_m':round(ci['c_wake']/1e6,4),'ohmic_mw':round(ci['ohmic_loss'],6),
                        'c_fatigue_m':round(ci.get('c_fatigue',0.0)/1e6,4),
                        'c_yaw_act_m':round(ci.get('c_yaw_act',0.0)/1e6,4),
                        'c_robust_m':round(ci.get('c_robust',0.0)/1e6,4),
                        'c_reliability_m':round(ci.get('c_reliability',0.0)/1e6,4),
                        'redundancy_ratio':round(ci.get('redundancy_ratio',0.0),4),
                        'n1_outage_turbines':ci.get('max_single_edge_outage_turbines',0),
                        'unserved_mwh':round(ci.get('expected_unserved_energy_mwh',0.0),4),
                        'crossing_violations':val_check.get('crossing_violations',0),
                        'cvar_power_mw':round(ci.get('cvar_power_mw',0.0),4),
                        'worst_power_mw':round(ci.get('worst_power_mw',0.0),4),
                        'worst_max_ti':round(ci.get('worst_max_ti',0.0),6),
                        'fatigue_damage_index':round(ci.get('fatigue_damage_index',0.0),6),
                        'max_ti_eff':round(ci.get('max_ti_eff',0.0),6),
                        'net_power_mw':round(ci['net_power'],4),
                        'mean_abs_yaw_deg':round(cm['mean_abs_yaw_deg'],4),
                        'gross_power_gain_mw':round(cm['gross_power_gain_mw'],4),
                        'ti_load_reduction':round(cm['ti_load_reduction'],6),
                        'fatigue_cost_saving_m':round(cm.get('fatigue_cost_saving',0.0)/1e6,4),
                        'pg_loss':round(self.agent.last_pg_loss,6),
                        'v_loss':round(self.agent.last_v_loss,6),
                        'entropy':round(self.agent.last_entropy,6),
                        'approx_kl':round(self.agent.last_approx_kl,6),
                    })

                    #  Periodic checkpoint every SAVE_INTERVAL episodes
                    if ep_cnt % TrainingConfig.SAVE_INTERVAL == 0 and ep_cnt > 0:
                        ckpt = os.path.join(CHECKPOINT_DIR, f"ckpt_ep{ep_cnt:05d}.pt")
                        torch.save({'actor_critic_state': self.agent.net.state_dict(),
                                    'episode': ep_cnt, 'stage': si,
                                    'min_cost': self.min_cost}, ckpt)
                        LOG.debug(f"  Periodic checkpoint → {ckpt}")

                    if ep_cnt%20==0:
                        r50=float(np.mean([c==0 for c in conn_hist[-50:]])) if conn_hist else 0.0
                        LOG.info(f"[PPO] ep={ep_cnt:4d} s={si} | rew={ep_rew:+8.1f} | "
                                 f"cost={cost_m:7.2f}M (inv={ci['c_inv']/1e6:.1f}, "
                                 f"op={ci['c_op']/1e6:.1f}, wk={ci['c_wake']/1e6:.1f}) | "
                                 f"iso={iso_cnt}/{self.env.num_turbines} | "
                                 f"succ50={r50:.2%} | KL={self.agent.last_approx_kl:.4f} | "
                                 f"H={self.agent.last_entropy:.3f}")
                    ep_cnt+=1; state=self.env.reset(use_mst_init=True)
                    ep_rew=0.0
                    if ep_cnt>=total_episodes: break

            # GAE computation [FIX-5 v2.0, v3-6]
            with torch.no_grad():
                _,_,_,next_val=self.agent.get_action(state,use_target=True,stage=self.curriculum.cur)
            raw_r=np.array(rew_b,dtype=np.float32)
            vals=np.array(val_b,dtype=np.float32)
            masks=np.array(mask_b,dtype=np.float32)
            advs=np.zeros(len(rew_b),dtype=np.float32); gae=0.0
            for t in reversed(range(len(rew_b))):
                # [v3-6] Use next_val only at end of buffer; otherwise use vals[t+1]
                if t==len(rew_b)-1:
                    nv=next_val
                else:
                    nv=vals[t+1]*masks[t+1]   # zero out nv if next step was terminal
                nm=masks[t]
                dt=raw_r[t]+PPOConfig.GAMMA*nv*nm-vals[t]
                gae=dt+PPOConfig.GAMMA*PPOConfig.GAE_LAMBDA*nm*gae
                advs[t]=gae
            rets=advs+vals
            rollout={'obs':obs_b,'actions':act_b,'yaw_actions':yaw_b,'logprobs':lp_b,'advantages':advs,
                     'returns':rets,'values':val_b,'valid_masks':vm_b}
            self.agent.update(rollout, stage=self.curriculum.cur)
            obs_b,act_b,yaw_b,lp_b,rew_b,val_b,mask_b,vm_b=[],[],[],[],[],[],[],[]

        return (rewards_hist, self.best_topo, self.best_cost_topo,
                self.best_reliable_topo, self.best_near_reliable_topo)


# ============================================================================
# SECTION 15b: POST-PPO GREEDY EDGE-SWAP REFINEMENT
# ============================================================================

def greedy_edge_swap_refinement(env, max_iters=200):
    return _base_shorter_edge_refinement(env, max_iters=max_iters)


def reliability_preserving_refinement(env, max_iters=80):
    """
    V10 clean-up: repair cable crossings first, then reduce cost, while keeping
    the topology inside the reliability gate. This prevents the final solution
    from being stripped back into a cheap radial tree.
    """
    LOG.info("  [v10] Running reliability-preserving crossing repair/refinement ...")
    initial_cost=env.calculate_lifecycle_cost()['total']/1e6
    initial_len=env._total_cable_length()/1000.0
    initial_val=env.validate_topology()
    initial_crossings=initial_val.get('crossing_violations',0)
    initial_rel=env.calculate_reliability_metrics()
    if not env._is_reliable_topology(initial_rel):
        LOG.info("  [v10] Current topology is not inside reliability gate; skipping refinement.")
        return {'initial_cost_m':round(initial_cost,4),'final_cost_m':round(initial_cost,4),
                'delta_cost_m':0.0,'iters_improved':0,
                'initial_cable_km':round(initial_len,4),'final_cable_km':round(initial_len,4),
                'initial_crossings':initial_crossings,'final_crossings':initial_crossings,
                'preserved_reliability':False}

    def crossing_count():
        return int(env.validate_topology().get('crossing_violations', 0))

    def snapshot():
        return (env.adjacency_matrix.copy(), env.edge_index.clone(), env.edge_attr.clone())

    def restore(snap):
        env.adjacency_matrix = snap[0].copy()
        env.edge_index = snap[1].clone()
        env.edge_attr = snap[2].clone()
        env._update_node_features()

    def feasible_reliable():
        rel=env.calculate_reliability_metrics()
        val=env.validate_topology()
        return (val['fully_connected'] and val['obstacle_violations']==0
                and env._is_reliable_topology(rel)), rel, val

    def repair_score(cost=None, val=None, rel=None):
        if cost is None: cost=env.calculate_lifecycle_cost()['total']/1e6
        if val is None: val=env.validate_topology()
        if rel is None: rel=env.calculate_reliability_metrics()
        cross=val.get('crossing_violations',0)
        n1=rel.get('max_single_edge_outage_turbines',env.num_turbines)
        ens=rel.get('expected_unserved_energy_mwh',1e9)
        red=rel.get('redundancy_ratio',0.0)
        return 12.0*cross + 0.15*cost + 0.25*n1 + 0.001*ens - 2.0*red

    improved=0
    best_score=repair_score(cost=initial_cost)
    for _ in range(max_iters):
        edges=env._get_unique_edges()
        if not edges: break
        current_cross=crossing_count()
        edge_items=[]
        for u,v in edges:
            line=LineString([env.node_positions[u],env.node_positions[v]])
            edge_cross=env._count_crossings(line, exclude_pair=(u,v))
            length=np.linalg.norm(env.node_positions[u]-env.node_positions[v])
            edge_items.append((edge_cross, length, u, v))
        edge_items.sort(reverse=True)
        accepted=False
        for old_cross, old_len, u, v in edge_items:
            if current_cross>ConstraintConfig.MAX_CROSSING_FOR_BEST and old_cross==0:
                continue
            snap=snapshot()

            # Option A: remove a redundant crossing edge if reliability survives.
            env._remove_edge(u,v)
            ok, rel, val=feasible_reliable()
            cost=env.calculate_lifecycle_cost()['total']/1e6
            score=repair_score(cost=cost, val=val, rel=rel)
            if ok and (score<best_score-1e-4 or
                       (current_cross>ConstraintConfig.MAX_CROSSING_FOR_BEST
                        and val.get('crossing_violations',0)<current_cross)):
                best_score=score; improved+=1; accepted=True
                break

            # Option B: replace it with a shorter/non-crossing alternative.
            restore(snap)
            env._remove_edge(u,v)
            best_alt=None; best_alt_score=best_score
            best_alt_cross=current_cross
            for i in range(env.num_nodes):
                for j in range(i+1,env.num_nodes):
                    if env.adjacency_matrix[i,j]>0: continue
                    d=np.linalg.norm(env.node_positions[i]-env.node_positions[j])
                    if d>=old_len and current_cross<=ConstraintConfig.MAX_CROSSING_FOR_BEST:
                        continue
                    line=LineString([env.node_positions[i],env.node_positions[j]])
                    if env._violates_obstacles(line): continue
                    if current_cross>ConstraintConfig.MAX_CROSSING_FOR_BEST:
                        new_cross=env._count_crossings(line, exclude_pair=(i,j))
                        if new_cross>=old_cross: continue
                    env._add_edge(i,j)
                    ok, rel, val=feasible_reliable()
                    cost=env.calculate_lifecycle_cost()['total']/1e6
                    env._remove_edge(i,j)
                    if not ok: continue
                    score=repair_score(cost=cost, val=val, rel=rel)
                    cross=val.get('crossing_violations',0)
                    improves_cross=(current_cross>ConstraintConfig.MAX_CROSSING_FOR_BEST
                                    and cross<best_alt_cross)
                    if score<best_alt_score-1e-4 or improves_cross:
                        best_alt=(i,j); best_alt_score=score; best_alt_cross=cross
            if best_alt is not None:
                env._add_edge(best_alt[0], best_alt[1])
                env._update_node_features()
                best_score=best_alt_score
                improved+=1; accepted=True
                break
            restore(snap)
        if not accepted: break

    final_cost=env.calculate_lifecycle_cost()['total']/1e6
    final_len=env._total_cable_length()/1000.0
    final_rel=env.calculate_reliability_metrics()
    final_val=env.validate_topology()
    return {'initial_cost_m':round(initial_cost,4),'final_cost_m':round(final_cost,4),
            'delta_cost_m':round(initial_cost-final_cost,4),'iters_improved':improved,
            'initial_cable_km':round(initial_len,4),'final_cable_km':round(final_len,4),
            'initial_crossings':initial_crossings,
            'final_crossings':final_val.get('crossing_violations',0),
            'preserved_reliability':env._is_reliable_topology(final_rel),
            'crossing_compliant':final_val.get('crossing_violations',0)<=ConstraintConfig.MAX_CROSSING_FOR_BEST}


def reliability_augmenting_refinement(env, max_additions=None, cleanup_iters=80):
    """
    V10 third fix: if PPO returns a near-reliable looped topology, add a small
    number of targeted redundant cables before cleanup. This is a deterministic
    feasibility repair, not another learned optimiser.
    """
    if max_additions is None:
        max_additions=ReliabilityConfig.AUGMENT_MAX_ADDITIONS
    LOG.info("  [v10] Running reliability-augmenting final repair ...")
    initial_cost=env.calculate_lifecycle_cost()['total']/1e6
    initial_len=env._total_cable_length()/1000.0
    initial_val=env.validate_topology()
    initial_rel=env.calculate_reliability_metrics()

    def snapshot():
        return (env.adjacency_matrix.copy(), env.edge_index.clone(), env.edge_attr.clone())

    def restore(snap):
        env.adjacency_matrix = snap[0].copy()
        env.edge_index = snap[1].clone()
        env.edge_attr = snap[2].clone()
        env._update_node_features()

    def reliability_gap_score(val=None, rel=None, cable_km=None):
        if val is None: val=env.validate_topology()
        if rel is None: rel=env.calculate_reliability_metrics()
        if cable_km is None: cable_km=env._total_cable_length()/1000.0
        if not val.get('fully_connected',False):
            return 1e9
        n1=rel.get('max_single_edge_outage_turbines', env.num_turbines)
        ens=rel.get('expected_unserved_energy_mwh', 1e9)
        red=rel.get('redundancy_ratio', 0.0)
        cross=val.get('crossing_violations',0)
        n1_gap=max(0, n1-ReliabilityConfig.RELIABLE_MAX_N1_OUTAGE_TURBINES)
        ens_gap=max(0.0, ens-ReliabilityConfig.RELIABLE_MAX_ENS_MWH)
        red_gap=max(0.0, ReliabilityConfig.RELIABLE_MIN_REDUNDANCY_RATIO-red)
        cross_gap=max(0, cross-ConstraintConfig.MAX_CROSSING_FOR_BEST)
        return (10000.0*val.get('obstacle_violations',0)
                + 180.0*n1_gap + 0.045*ens_gap + 320.0*red_gap
                + 18.0*cross_gap + 0.70*cross + 0.035*cable_km
                - 3.0*red)

    added=0
    current_score=reliability_gap_score(initial_val, initial_rel, initial_len)
    for _ in range(max_additions):
        val=env.validate_topology()
        rel=env.calculate_reliability_metrics()
        if env._is_reliable_topology(rel) and val.get('crossing_violations',0)<=ConstraintConfig.MAX_CROSSING_FOR_BEST:
            break
        candidates=[]
        for i in range(env.num_nodes):
            for j in range(i+1, env.num_nodes):
                if env.adjacency_matrix[i,j]>0:
                    continue
                line=LineString([env.node_positions[i], env.node_positions[j]])
                if env._violates_obstacles(line):
                    continue
                d=np.linalg.norm(env.node_positions[i]-env.node_positions[j])
                add_cross=env._count_crossings(line, exclude_pair=(i,j))
                candidates.append((add_cross, d, i, j))
        candidates.sort(key=lambda x: (x[0], x[1]))
        if ReliabilityConfig.AUGMENT_CANDIDATE_LIMIT:
            candidates=candidates[:ReliabilityConfig.AUGMENT_CANDIDATE_LIMIT]

        best_pair=None; best_score=current_score
        best_detail=None
        for add_cross, _, i, j in candidates:
            snap=snapshot()
            env._add_edge(i,j)
            env._update_node_features()
            val=env.validate_topology()
            rel=env.calculate_reliability_metrics()
            cable_km=env._total_cable_length()/1000.0
            score=reliability_gap_score(val, rel, cable_km)
            restore(snap)
            if score<best_score-1e-6:
                best_score=score; best_pair=(i,j)
                best_detail=(rel.get('redundancy_ratio',0.0),
                             rel.get('max_single_edge_outage_turbines',env.num_turbines),
                             rel.get('expected_unserved_energy_mwh',0.0),
                             val.get('crossing_violations',0), add_cross)

        if best_pair is None:
            LOG.info("  [v10] No reliability-augmenting edge can improve the repair score.")
            break
        env._add_edge(best_pair[0], best_pair[1])
        env._update_node_features()
        current_score=best_score
        added+=1
        if best_detail:
            LOG.info(f"  [v10] Added reliability edge {best_pair[0]}-{best_pair[1]} | "
                     f"red={best_detail[0]:.3f}, N1={best_detail[1]}, "
                     f"ENS={best_detail[2]:.1f}, cross={best_detail[3]}")

    pre_cleanup_val=env.validate_topology()
    pre_cleanup_rel=env.calculate_reliability_metrics()
    cleanup_stats={}
    if env._is_reliable_topology(pre_cleanup_rel):
        cleanup_stats=reliability_preserving_refinement(env, max_iters=cleanup_iters)

    final_cost=env.calculate_lifecycle_cost()['total']/1e6
    final_len=env._total_cable_length()/1000.0
    final_val=env.validate_topology()
    final_rel=env.calculate_reliability_metrics()
    return {'initial_cost_m':round(initial_cost,4),
            'final_cost_m':round(final_cost,4),
            'delta_cost_m':round(initial_cost-final_cost,4),
            'edges_added':added,
            'initial_cable_km':round(initial_len,4),
            'final_cable_km':round(final_len,4),
            'initial_crossings':initial_val.get('crossing_violations',0),
            'pre_cleanup_crossings':pre_cleanup_val.get('crossing_violations',0),
            'final_crossings':final_val.get('crossing_violations',0),
            'initial_redundancy_ratio':round(initial_rel.get('redundancy_ratio',0.0),4),
            'final_redundancy_ratio':round(final_rel.get('redundancy_ratio',0.0),4),
            'final_n1_outage_turbines':final_rel.get('max_single_edge_outage_turbines',env.num_turbines),
            'final_expected_ens_mwh':round(final_rel.get('expected_unserved_energy_mwh',0.0),4),
            'preserved_reliability':env._is_reliable_topology(final_rel),
            'crossing_compliant':final_val.get('crossing_violations',0)<=ConstraintConfig.MAX_CROSSING_FOR_BEST,
            'cleanup':cleanup_stats}


# ============================================================================
# SECTION 16: RESULTS EXPORT
# ============================================================================

def export_results_json(comparison_jensen, comparison_cfd, layout_metrics,
                        pretrain_stats, episode_log, best_topo_info,
                        final_validation, final_cost, env, stage_log,
                        refinement_stats=None, reliability_candidates=None):
    save_json({'description':'Algorithm convergence histories (power MW per iteration)',
               'jensen':{k:v for k,v in comparison_jensen.items() if not k.startswith('_')},
               'cfd'   :{k:v for k,v in comparison_cfd.items()    if not k.startswith('_')}},
              "fig1_convergence_curves.json")

    fig2_rows=[]
    common=(set(comparison_jensen.keys())&set(comparison_cfd.keys()))-{'_cfd_stats'}
    for algo in sorted(common):
        pj=comparison_jensen[algo]['power_mw']; pc=comparison_cfd[algo]['power_mw']
        fig2_rows.append({'algorithm':algo,'power_jensen_mw':round(pj,4),
                          'power_cfd_mw':round(pc,4),
                          'improvement_pct':round((pc-pj)/(pj+1e-8)*100,4),
                          'evals_jensen':comparison_jensen[algo]['n_evals'],
                          'evals_cfd':comparison_cfd[algo]['n_evals'],
                          'time_s_jensen':comparison_jensen[algo]['time_s'],
                          'time_s_cfd':comparison_cfd[algo]['time_s']})
    save_json({'description':'Jensen vs CFD final power comparison','rows':fig2_rows},
              "fig2_model_comparison_table.json")

    save_json({'description':'Final NSGA-II layout metrics [I2]',**layout_metrics},
              "fig4_layout_metrics.json")
    ppo_sample=episode_log[::5] if len(episode_log)>5 else episode_log
    save_json({'description':'PPO episode learning curves','episodes':ppo_sample},
              "fig6_ppo_learning_curves.json")
    save_json({'description':'Curriculum stage transition log','transitions':stage_log},
              "fig7_curriculum_stages.json")
    save_json({'description':'Reliability-aware candidate archive',
               'selection_gate':{
                   'max_n1_outage_turbines':ReliabilityConfig.RELIABLE_MAX_N1_OUTAGE_TURBINES,
                   'min_redundancy_ratio':ReliabilityConfig.RELIABLE_MIN_REDUNDANCY_RATIO,
                   'max_expected_unserved_mwh':ReliabilityConfig.RELIABLE_MAX_ENS_MWH,
                   'near_max_n1_outage_turbines':ReliabilityConfig.NEAR_RELIABLE_MAX_N1_OUTAGE_TURBINES,
                   'near_min_redundancy_ratio':ReliabilityConfig.NEAR_RELIABLE_MIN_REDUNDANCY_RATIO,
                   'near_max_expected_unserved_mwh':ReliabilityConfig.NEAR_RELIABLE_MAX_ENS_MWH,
               },
               'candidates':reliability_candidates or []},
              "fig9_reliability_candidates.json")

    node_data=[]; W,H=EnvironmentConfig.GRID_SIZE
    for i in range(env.num_nodes):
        row={'id':i,'x':round(float(env.node_positions[i,0]),2),
             'y':round(float(env.node_positions[i,1]),2),
             'type':'substation' if i>=env.num_turbines else 'turbine'}
        if i<env.num_turbines:
            row['yaw_deg']=round(float(env.yaw_angles[i]),4)
            row['yaw_power_factor']=round(float(env.wake_cache.get('yaw_power_factor', np.ones(env.num_turbines))[i]),6)
        node_data.append(row)
    edges_data=[{'u':int(u),'v':int(v)} for u,v in env._get_unique_edges()]
    save_json({'description':'Final topology graph','nodes':node_data,'edges':edges_data,
               'validation':final_validation,'cost':final_cost}, "fig8_final_topology.json")
    save_json({'description':'Paper summary metrics','best_topology':best_topo_info,
               'layout':layout_metrics,'pretrain':pretrain_stats,
               'n_episodes':len(episode_log),'stage_transitions':stage_log,
               'refinement':refinement_stats or {},
               'reliability_candidates':reliability_candidates or []},
              "paper_summary_table.json")


# ============================================================================
# SECTION 17: MAIN PIPELINE
# ============================================================================

def main():
    _base_write_metadata()
    t_global=time.perf_counter()
    device=torch.device(DeviceConfig.DEVICE)

    log_section("CFD-GNN-PPO Reliability-Aware Robust Tri-Level Framework  v10.0")
    LOG.info(f"Device={device}  Seed={RANDOM_SEED}")
    LOG.info(f"Grid={EnvironmentConfig.GRID_SIZE} m  Turbines={EnvironmentConfig.NUM_TURBINES}  "
             f"Rated={EnvironmentConfig.RATED_POWER} MW  D={EnvironmentConfig.ROTOR_DIAMETER} m (Vestas V164-8.0 MW)")
    LOG.info(f"v10 key changes: reliability-aware cable topology, N-1 outage exposure, "
             f"expected unserved energy, redundancy reward")

    n_turbines=EnvironmentConfig.NUM_TURBINES
    domain_bounds={'x':(0,EnvironmentConfig.GRID_SIZE[0]),'y':(0,EnvironmentConfig.GRID_SIZE[1])}

    results_jensen, results_cfd, best_layout_cfd, avg_imp = \
        run_baseline_comparison(n_turbines, domain_bounds)

    env=OffshoreWindEnv(initial_layout=best_layout_cfd)
    nsga_optimizer=HybridAICFDLayoutOptimizer(env)
    best_layout=nsga_optimizer.optimize_layout()

    env=OffshoreWindEnv(initial_layout=best_layout)
    feasible_fallback = _base_preflight(env)
    layout_metrics={
        'power_mw'       : round(nsga_optimizer.final_power,4),
        'wake_loss_ratio': round(nsga_optimizer.final_wake_ratio,5),
        'load_index'     : round(nsga_optimizer.final_load_index,5),
        'cable_cost_m'   : round(nsga_optimizer.cable_cost/1e6,4),
    }
    LOG.info(f"  NSGA-II | power={layout_metrics['power_mw']:.3f} MW | "
             f"wake_loss={layout_metrics['wake_loss_ratio']:.4f} | "
             f"TI_load_idx={layout_metrics['load_index']:.5f}")

    mapper   =ActionMapper(env.num_turbines, EnvironmentConfig.NUM_SUBSTATIONS,
                            env.node_positions)
    action_dim=mapper.get_action_space_size()

    gnn_pre=GNNEncoder(node_feature_dim=GNNConfig.NODE_FEATURE_DIM,
                        hidden_dim=GNNConfig.HIDDEN_DIM,
                        output_dim=GNNConfig.OUTPUT_DIM).to(device)

    data_gen=TopologyDataGenerator(env)
    samples=data_gen.generate(DataGenerationConfig.NUM_SAMPLES, best_layout, mapper)

    pretrainer=MultiTaskPretrainer(gnn_pre, device, action_dim)
    pretrain_stats=pretrainer.train(samples, GNNConfig.PRETRAIN_EPOCHS, mapper, env)


    gnn_ac=GNNEncoder(node_feature_dim=GNNConfig.NODE_FEATURE_DIM,
                       hidden_dim=GNNConfig.HIDDEN_DIM, output_dim=GNNConfig.OUTPUT_DIM)
    gnn_ac.load_state_dict(gnn_pre.state_dict())
    actor_critic=GNNActorCritic(gnn_ac, action_dim).to(device)

    # Warm-start actor from pre-trained action_logits head
    with torch.no_grad():
        pt_head=pretrainer.heads['action_logits'].weight.data
        ac_layer=actor_critic.actor[-1].weight.data
        min_rows=min(ac_layer.shape[0],pt_head.shape[0])
        min_cols=min(ac_layer.shape[1],pt_head.shape[1])
        actor_critic.actor[-1].weight.data[:min_rows,:min_cols]=pt_head[:min_rows,:min_cols]
        LOG.info(f"  Actor warm-start: transferred {min_rows}×{min_cols} weights")

    agent  =PPOAgent(actor_critic, device, total_episodes=TrainingConfig.NUM_EPISODES)
    trainer=PPOTrainer(env, agent, mapper)
    (rewards_hist, best_topo, best_cost_topo,
     best_reliable_topo, best_near_reliable_topo)=trainer.train(TrainingConfig.NUM_EPISODES)

    log_section("Phase 4 – Final Evaluation & JSON Export")

    best_topo_info={}; final_validation={}; final_cost={}; refinement_stats={}; final_control={}

    active_topo = best_reliable_topo or best_near_reliable_topo or best_topo
    if best_reliable_topo is not None:
        rel=best_reliable_topo.get('reliability',{})
        LOG.info(f"  [v10] Selecting reliability-aware compromise topology "
                 f"(ep={best_reliable_topo['episode']}, cost={best_reliable_topo['cost_m']:.3f}M, "
                 f"red={rel.get('redundancy_ratio',0.0):.3f}, "
                 f"N1={rel.get('max_single_edge_outage_turbines','-')}, "
                 f"ENS={rel.get('expected_unserved_energy_mwh',0.0):.1f} MWh)")
    elif best_near_reliable_topo is not None:
        rel=best_near_reliable_topo.get('reliability',{})
        LOG.info(f"  [v10] Selecting near-reliable topology for deterministic augmentation "
                 f"(ep={best_near_reliable_topo['episode']}, "
                 f"cost={best_near_reliable_topo['cost_m']:.3f}M, "
                 f"red={rel.get('redundancy_ratio',0.0):.3f}, "
                 f"N1={rel.get('max_single_edge_outage_turbines','-')}, "
                 f"ENS={rel.get('expected_unserved_energy_mwh',0.0):.1f} MWh)")
    if active_topo is None and best_cost_topo is not None:
        LOG.warning("  No crossing-compliant topology found; using best-cost topology "
                    f"(cross={best_cost_topo['crossing_violations']})")
        active_topo = best_cost_topo
    elif (active_topo is not None and best_cost_topo is not None
          and best_cost_topo['cost_m'] < active_topo['cost_m'] - 0.01):
        LOG.info(f"  [v6-3] best_cost_topo ({best_cost_topo['cost_m']:.3f}M) is cheaper than "
                 f"best_valid_topo ({active_topo['cost_m']:.3f}M); both reported.")

    if active_topo is None:
        active_topo = feasible_fallback
    if active_topo is not None:
        LOG.info(f"  Restoring best topology (ep={active_topo['episode']}, "
                 f"cost={active_topo['cost_m']:.3f}M)")
        active_topo, refinement_stats = _base_finalize(env, active_topo, feasible_fallback)

        final_cost      =env.calculate_lifecycle_cost()
        final_validation=env.validate_topology()
        final_control   =env.calculate_control_metrics()

        fc={k:(round(float(v),6) if isinstance(v,(float,np.floating)) else v)
             for k,v in final_cost.items()}
        fv={k:(int(v) if isinstance(v,(int,np.integer)) else v)
             for k,v in final_validation.items()}

        LOG.info("="*60)
        LOG.info("  FINAL TOPOLOGY VALIDATION")
        LOG.info(f"    Fully Connected   : {final_validation['fully_connected']}")
        LOG.info(f"    Isolated Count    : {final_validation['isolated_count']}")
        LOG.info(f"    Obstacle Viol     : {final_validation['obstacle_violations']}")
        LOG.info(f"    Crossing Viol     : {final_validation['crossing_violations']} "
                 f"(threshold ≤{ConstraintConfig.MAX_CROSSING_FOR_BEST})")
        LOG.info(f"    Is Valid [v6-1]   : {final_validation['is_valid']}")
        LOG.info("  25-YEAR LIFECYCLE COST BREAKDOWN  [I5]")
        LOG.info(f"    C_inv  (CAPEX)    : {final_cost['c_inv']/1e6:.3f} M RMB")
        LOG.info(f"    C_op   (OPEX)     : {final_cost['c_op']/1e6:.3f} M RMB")
        LOG.info(f"    C_wake (opp cost) : {final_cost['c_wake']/1e6:.3f} M RMB")
        LOG.info(f"    C_fatigue         : {final_cost.get('c_fatigue',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_yaw_act         : {final_cost.get('c_yaw_act',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_robust          : {final_cost.get('c_robust',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_reliability     : {final_cost.get('c_reliability',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_total           : {final_cost['total']/1e6:.3f} M RMB")
        LOG.info("  POWER STATISTICS")
        LOG.info(f"    Deliverable Power : {final_cost['deliverable_power']:.4f} MW")
        LOG.info(f"    Ohmic Loss        : {final_cost['ohmic_loss']:.6f} MW")
        LOG.info(f"    Net Power         : {final_cost['net_power']:.4f} MW")
        LOG.info(f"    CVaR Power        : {final_cost.get('cvar_power_mw',final_cost['net_power']):.4f} MW")
        LOG.info(f"    Worst Power       : {final_cost.get('worst_power_mw',final_cost['net_power']):.4f} MW")
        LOG.info(f"    Connected Turbines: {final_cost['connected_turbines']}/{env.num_turbines}")
        LOG.info("  CABLE RELIABILITY  [I10]")
        LOG.info(f"    Redundancy Ratio  : {final_cost.get('redundancy_ratio',0.0):.4f}")
        LOG.info(f"    N-1 Max Outage    : {final_cost.get('max_single_edge_outage_turbines',0)} turbines")
        LOG.info(f"    Expected ENS      : {final_cost.get('expected_unserved_energy_mwh',0.0):.3f} MWh/yr")
        LOG.info("  ACTIVE YAW CONTROL  [I7]")
        LOG.info(f"    Mean |Yaw|        : {final_control['mean_abs_yaw_deg']:.3f} deg")
        LOG.info(f"    Gross Power Gain  : {final_control['gross_power_gain_mw']:+.4f} MW")
        LOG.info(f"    TI-load Reduction : {final_control['ti_load_reduction']:+.6f}")
        LOG.info(f"    Fatigue Saving    : {final_control.get('fatigue_cost_saving',0.0)/1e6:+.3f} M RMB")
        LOG.info(f"    Damage Reduction  : {final_control.get('fatigue_damage_reduction',0.0):+.6f}")
        LOG.info("="*60)

        best_topo_info={
            'best_episode' : active_topo['episode'],
            'cost_m_pre_refine' : round(active_topo['cost_m'],4),
            'cost_m_post_refine': round(final_cost['total']/1e6,4),
            'n_edges'      : len(env._get_unique_edges()),
            'total_cable_km': round(env._total_cable_length()/1000.0,4),
            'refinement'   : refinement_stats,
            'selection_tag': active_topo.get('tag','unknown'),
            'selection_score': round(float(active_topo.get('selection_score', 0.0)),6),
            'yaw_control'  : {k:round(float(v),6) for k,v in final_control.items()},
            'robust'       : {k:round(float(final_cost.get(k,0.0)),6)
                              for k in ['c_robust','c_robust_power','c_robust_ti',
                                        'expected_power_mw','cvar_power_mw','worst_power_mw',
                                        'expected_max_ti','worst_max_ti']},
            'reliability'  : {k:round(float(final_cost.get(k,0.0)),6)
                              for k in ['c_reliability','c_unserved_energy','c_n1_exposure',
                                        'expected_unserved_energy_mwh','max_single_edge_outage_turbines',
                                        'n_bridge_edges','n_cycle_edges','redundancy_ratio']},
        }
        if best_cost_topo is not None:
            best_topo_info['best_cost_unrestricted_m'] = round(best_cost_topo['cost_m'],4)
            best_topo_info['best_cost_unrestricted_ep'] = best_cost_topo['episode']
        if best_reliable_topo is not None:
            best_topo_info['best_reliable_m'] = round(best_reliable_topo['cost_m'],4)
            best_topo_info['best_reliable_ep'] = best_reliable_topo['episode']
            best_topo_info['best_reliable_selection_score'] = round(
                float(best_reliable_topo.get('selection_score',0.0)),4)
        if best_near_reliable_topo is not None:
            best_topo_info['best_near_reliable_m'] = round(best_near_reliable_topo['cost_m'],4)
            best_topo_info['best_near_reliable_ep'] = best_near_reliable_topo['episode']
            best_topo_info['best_near_reliable_selection_score'] = round(
                float(best_near_reliable_topo.get('selection_score',0.0)),4)

        ckpt_path=os.path.join(CHECKPOINT_DIR,"best_actor_critic_v10_reliability.pt")
        torch.save({'actor_critic_state':actor_critic.state_dict(),
                    'best_episode':active_topo['episode'],'cost_m':active_topo['cost_m'],
                    'yaw_angles':env.yaw_angles,
                    'version':'v10.0','innovations':['I1-Frandsen','I2-NSGA-II','I3-GATv2',
                                                    'I4-PPO-IQN','I5-Lifecycle','I6-Benchmark',
                                                    'I7-Active-Yaw-Wake-Steering',
                                                    'I8-Fatigue-Economics',
                                                    'I9-Robust-Uncertainty',
                                                    'I10-Cable-Reliability'],
                    'v10_changes':['expected-unserved-energy','n-1-outage-exposure',
                                   'redundancy-ratio','reliability-cost-reward',
                                   'reliable-candidate-selection',
                                   'near-reliable-candidate-archive',
                                   'reliability-preserving-refinement',
                                   'reliability-augmenting-final-repair']},
                   ckpt_path)
        LOG.info(f"  Checkpoint saved → {ckpt_path}")

        # Turbine export with Frandsen TI
        env._precompute_wake()
        turbine_sheet=[]
        for i in range(env.num_turbines):
            v_eff=env.wake_cache['effective_velocities'][i]
            ti_eff=env.wake_cache['ti_effective'][i]
            yaw_pf=env.wake_cache.get('yaw_power_factor', np.ones(env.num_turbines))[i]
            R=EnvironmentConfig.ROTOR_DIAMETER/2.0
            hc=(EnvironmentConfig.Z_HUB/EnvironmentConfig.Z_REF)**EnvironmentConfig.ABL_ALPHA
            p=min(0.5*EnvironmentConfig.AIR_DENSITY*math.pi*R**2*(v_eff*hc)**3
                  *EnvironmentConfig.Cp*EnvironmentConfig.ETA_GEAR*yaw_pf/1e6, EnvironmentConfig.RATED_POWER)
            turbine_sheet.append({'Turbine_ID':i,'X_coord':round(float(env.node_positions[i,0]),2),
                                   'Y_coord':round(float(env.node_positions[i,1]),2),
                                   'Yaw_deg':round(float(env.yaw_angles[i]),4),
                                   'Yaw_power_factor':round(float(yaw_pf),6),
                                   'Eff_Vel_ms':round(float(v_eff),4),'Power_MW':round(float(p),6),
                                   'TI_Frandsen':round(float(ti_eff),5),
                                   'Connected':i in env._get_connected_nodes()})
        save_json({'Turbines':turbine_sheet,
                   'Metadata':{'Algorithm':'CFD-GNN-PPO Reliability Robust Tri-Level v10.0',
                                'Total_Power_MW':round(float(final_cost['net_power']),4),
                                'N_Turbines':env.num_turbines,'Version':'v10.0',
                                'Key_Changes':['Frandsen-TI-quadrature-v5','18-dim-yaw-features',
                                               'cosine-LR','crossing-threshold-0',
                                               'unified-turbine-params',
                                               'entropy-PID','entropy-refresh-v6',
                                               'dynamic-cost-reward','greedy-edge-swap-v6',
                                               'secondary-best-tracker-v6',
                                               'active-yaw-wake-steering-v7',
                                               '18-dim-yaw-state',
                                               'hybrid-discrete-continuous-PPO',
                                               'fatigue-economics-v8',
                              'robust-uncertainty-v9',
                              'cable-reliability-v10',
                              'reliable-topology-selection-fix',
                              'near-reliable-augmentation-fix']}},
                  "export_optimised_layout.json")
    else:
        LOG.warning("  No valid topology found during PPO training.")
        fv={}; fc={}

    export_results_json(
        comparison_jensen=results_jensen, comparison_cfd=results_cfd,
        layout_metrics=layout_metrics, pretrain_stats=pretrain_stats,
        episode_log=trainer.episode_log, best_topo_info=best_topo_info,
        final_validation=fv, final_cost=fc, env=env,
        stage_log=trainer.curriculum.get_stage_transitions(),
        refinement_stats=refinement_stats,
        reliability_candidates=trainer.reliability_candidates)

    elapsed=time.perf_counter()-t_global
    LOG.info("="*72)
    LOG.info(f"  ALL PHASES COMPLETE | total_wall_time={elapsed:.1f}s")
    LOG.info(f"  Results → {RESULTS_DIR}/  |  Plot data → {PLOT_DATA_DIR}/")
    LOG.info("="*72)
    if avg_imp is not None:
        LOG.info(f"  ║  CFD vs Jensen AEP improvement     : {avg_imp:+6.2f}%           ║")
    LOG.info(f"  ║  NSGA-II layout power              : {layout_metrics['power_mw']:7.3f} MW         ║")
    LOG.info(f"  ║  NSGA-II wake loss ratio           : {layout_metrics['wake_loss_ratio']:7.4f}            ║")
    LOG.info(f"  ║  NSGA-II TI load index (Frandsen)  : {layout_metrics['load_index']:7.5f}            ║")
    LOG.info(f"  ║  GNN pre-train action accuracy     : {pretrain_stats.get('action_acc',0.0):7.4f}            ║")
    if final_cost:
        LOG.info(f"  ║  Final lifecycle cost (25 yr)      : {final_cost['total']/1e6:7.3f} M RMB    ║")
        LOG.info(f"  ║  Net deliverable power             : {final_cost['net_power']:7.4f} MW         ║")
    if refinement_stats:
        LOG.info(f"  ║  Greedy refinement Δcost [v6-5]   : {refinement_stats.get('delta_cost_m',0.0):+7.3f} M RMB    ║")
    if best_topo_info.get('best_cost_unrestricted_m'):
        LOG.info(f"  ║  Best cost (unrestricted) [v6-3]  : {best_topo_info['best_cost_unrestricted_m']:7.3f} M RMB    ║")
    if final_validation:
        lbl='YES' if final_validation.get('is_valid') else 'NO '
        crs=final_validation.get('crossing_violations','-')
        LOG.info(f"  ║  Topology valid (cross≤{ConstraintConfig.MAX_CROSSING_FOR_BEST})          : {lbl} (cross={crs})          ║")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
