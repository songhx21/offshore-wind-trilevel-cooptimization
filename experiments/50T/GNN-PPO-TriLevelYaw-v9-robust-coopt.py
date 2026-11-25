
import os, sys, json, math, time, random, logging, warnings
from collections import defaultdict, deque
from performance_cache import (
    caches_enabled, layout_cache_key, cached_layout_objectives,
    cached_wake_evaluation, cached_environment_method,
    cached_line, cached_line_conflict, cached_obstacle_polygon,
    cached_baseline_comparison, cache_reported_main,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

import networkx as nx
from shapely.geometry import Point, LineString
from large_case_config import (
    CASE_ID,
    CASE_NAME,
    COLLECTION_VOLTAGE_KV as LARGE_CASE_COLLECTION_VOLTAGE_KV,
    FORBIDDEN_ZONES as LARGE_CASE_FORBIDDEN_ZONES,
    GRID_SIZE as LARGE_CASE_GRID_SIZE,
    HUB_HEIGHT_M as LARGE_CASE_HUB_HEIGHT_M,
    MAX_ALLOWED_CROSSINGS as LARGE_CASE_MAX_ALLOWED_CROSSINGS,
    MAX_CABLE_LENGTH_M as LARGE_CASE_MAX_CABLE_LENGTH_M,
    MAX_TOPOLOGY_STEPS as LARGE_CASE_MAX_TOPOLOGY_STEPS,
    MIN_SPACING_FACTOR as LARGE_CASE_MIN_SPACING_FACTOR,
    NUM_SUBSTATIONS as LARGE_CASE_NUM_SUBSTATIONS,
    NUM_TURBINES as LARGE_CASE_NUM_TURBINES,
    RATED_POWER_MW as LARGE_CASE_RATED_POWER_MW,
    ROTOR_DIAMETER_M as LARGE_CASE_ROTOR_DIAMETER_M,
    SUBSTATION_POSITIONS as LARGE_CASE_SUBSTATION_POSITIONS,
    build_wind_scenarios,
    case_metadata,
    count_site_violations,
    dominant_wind_scenario,
    is_large_case_layout_feasible,
    layout_violation_score,
    project_points_to_site,
    repair_large_case_layout,
    sample_large_case_layout,
    write_case_definition,
)

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
# variable. Every launch receives a new run directory, so existing results are
# never overwritten, even when the same seed is repeated.
DEFAULT_RANDOM_SEED = 133
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

RESULTS_ROOT = os.path.join("results", "iea37_cs4_50t", "cfd_gnn_ppo_trilevel_yaw_v9_robust_coopt")

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
write_case_definition(os.path.join(PLOT_DATA_DIR, "large_case_definition.json"))

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
    BOUNDARY_PADDING       = 0.0
    MAX_CROSSING_FOR_BEST  = LARGE_CASE_MAX_ALLOWED_CROSSINGS

class EnvironmentConfig:
    NUM_TURBINES                = LARGE_CASE_NUM_TURBINES
    NUM_SUBSTATIONS             = LARGE_CASE_NUM_SUBSTATIONS
    GRID_SIZE                   = LARGE_CASE_GRID_SIZE
    MAX_STEPS                   = LARGE_CASE_MAX_TOPOLOGY_STEPS
    RATED_POWER                 = LARGE_CASE_RATED_POWER_MW
    ROTOR_DIAMETER              = LARGE_CASE_ROTOR_DIAMETER_M
    HUB_HEIGHT                  = LARGE_CASE_HUB_HEIGHT_M
    MIN_TURBINE_DISTANCE_FACTOR = LARGE_CASE_MIN_SPACING_FACTOR
    LAYOUT_REPAIR_ITERS         = 90
    LAYOUT_REPAIR_ATTEMPTS      = 40
    LAYOUT_CVAR_ALPHA           = 0.20
    LAYOUT_POWER_STD_WEIGHT     = 0.25
    LAYOUT_FATIGUE_DAMAGE_WEIGHT = 0.10
    LAYOUT_TOPOLOGY_FEEDBACK     = True
    LAYOUT_TOPOLOGY_EXTRA_EDGE_RATIO = 0.16
    LAYOUT_TOPOLOGY_MIN_DEGREE   = 2
    LAYOUT_N1_RISK_COST_KM       = 20.0
    LAYOUT_CROSSING_COST_KM      = 5.0
    LAYOUT_TOPOLOGY_RISK_MW_PENALTY = 8.0
    AIR_DENSITY                 = 1.225
    Cp                          = 0.48
    ETA_GEAR                    = 0.95
    # Wake parameters
    WAKE_DECAY_CONSTANT         = 0.0324555
    THRUST_COEFFICIENT          = 8.0 / 9.0
    INDUCTION_FACTOR            = 1 - math.sqrt(1 - 8.0 / 9.0)
    TI_AMBIENT                  = 0.075
    ABL_ALPHA                   = 0.12
    Z_HUB                       = LARGE_CASE_HUB_HEIGHT_M
    Z_REF                       = LARGE_CASE_HUB_HEIGHT_M
    # Active wake steering / yaw-control layer [v7]
    YAW_MAX_DEG                 = 25.0         # bounded continuous control signal
    YAW_POWER_EXPONENT          = 1.88         # P_yaw = P_0 cos(gamma)^alpha
    YAW_CT_EXPONENT             = 2.0          # C_T,yaw = C_T cos(gamma)^2
    WAKE_STEERING_GAIN          = 0.30         # empirical BPA centreline deflection gain
    YAW_TI_RELIEF_GAIN          = 0.35         # down-weight added TI from yawed wakes
    YAW_CONTROL_REWARD_WEIGHT   = 3000.0       # reward scale for TI relief and yaw effort
    YAW_EFFORT_PENALTY          = 0.015        # soft penalty per mean |yaw| / yaw_max
    # Electrical
    COLLECTION_VOLTAGE          = LARGE_CASE_COLLECTION_VOLTAGE_KV
    CABLE_RESISTANCE_PER_METER  = 0.00015
    POWER_FACTOR                = 0.95
    MAX_CABLE_LENGTH            = LARGE_CASE_MAX_CABLE_LENGTH_M
    FORBIDDEN_ZONES = [dict(zone) for zone in LARGE_CASE_FORBIDDEN_ZONES]
class LifecycleConfig:
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
    FATIGUE_ROUTE_COST_PER_WEIGHTED_KM = 1_200_000.0
    FATIGUE_CRITICAL_ROUTE_COST_PER_KM = 240_000.0

class RobustConfig:
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
    # Re-optimisation layer: robust-aware electrical routing objective.
    # Turbines with high uncertainty vulnerability are penalised for long
    # paths to the substation, allowing v9 to choose a different topology.
    ROBUST_ROUTE_COST_PER_WEIGHTED_KM = 1_350_000.0
    ROBUST_LOW_POWER_ROUTE_COST_PER_KM = 300_000.0

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

    @cached_wake_evaluation
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
        self.min_spacing = (EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR
                            * sim.D)

    def _fitness(self, x, y):
        penalty = 0.0
        for i in range(self.n_turbines):
            for j in range(i+1, self.n_turbines):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if d < self.min_spacing:
                    penalty += (self.min_spacing-d)**2
        return self.sim.compute_power(x, y) - 1e5*penalty

    def _random_positions(self):
        layout = sample_large_case_layout(
            n_turbines=self.n_turbines,
            min_spacing=(EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR
                         * EnvironmentConfig.ROTOR_DIAMETER),
            forbidden_zones=EnvironmentConfig.FORBIDDEN_ZONES,
            obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN,
            rng=np.random,
        )
        return layout[:, 0].copy(), layout[:, 1].copy()

    def _clip(self, x, y):
        layout = np.column_stack([np.asarray(x, dtype=float),
                                  np.asarray(y, dtype=float)])
        layout = repair_large_case_layout(
            layout,
            n_turbines=self.n_turbines,
            min_spacing=(EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR
                         * EnvironmentConfig.ROTOR_DIAMETER),
            forbidden_zones=EnvironmentConfig.FORBIDDEN_ZONES,
            obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN,
            rng=np.random,
            iterations=80,
        )
        return layout[:, 0].copy(), layout[:, 1].copy()


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
        pos=[self._random_positions() for _ in range(self.n_particles)]
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

@cached_baseline_comparison
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
        dominant_wind = dominant_wind_scenario()
        self.wind_speed = float(dominant_wind['wind_speed'])
        self.wind_direction = float(dominant_wind['wind_direction'])
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
        rng = np.random.RandomState(RANDOM_SEED)
        min_spacing = (EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR
                       * EnvironmentConfig.ROTOR_DIAMETER)
        if layout is not None:
            positions = repair_large_case_layout(
                np.asarray(layout[:self.num_turbines], dtype=float),
                n_turbines=self.num_turbines,
                min_spacing=min_spacing,
                forbidden_zones=EnvironmentConfig.FORBIDDEN_ZONES,
                obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN,
                rng=rng,
                iterations=120,
            )
        else:
            positions = sample_large_case_layout(
                n_turbines=self.num_turbines,
                min_spacing=min_spacing,
                forbidden_zones=EnvironmentConfig.FORBIDDEN_ZONES,
                obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN,
                rng=rng,
            )
        sub_pos = LARGE_CASE_SUBSTATION_POSITIONS.copy()
        self.node_positions = np.vstack([positions, sub_pos])
        self.turbine_positions = self.node_positions[:self.num_turbines]
        self.substation_positions = self.node_positions[self.num_turbines:]
        self._clamp_turbines()

    def _clamp_turbines(self):
        projected = project_points_to_site(
            self.node_positions[:self.num_turbines]
        )
        self.node_positions[:self.num_turbines] = projected
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

    @cached_environment_method("topology")
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

    @cached_environment_method("geometry")
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

    @cached_environment_method("geometry")
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

    @cached_environment_method("topology")
    def _crossing_feature(self):
        """Fraction of edges that cross at least one other edge (per-node proxy)."""
        edges=self._get_unique_edges(); N=len(edges); cross_nodes=np.zeros(self.num_nodes)
        if N==0: return cross_nodes
        for k,(u1,v1) in enumerate(edges):
            l1=cached_line(self.node_positions[u1], self.node_positions[v1])
            for (u2,v2) in edges[k+1:]:
                if {u1,v1}&{u2,v2}: continue
                l2=cached_line(self.node_positions[u2], self.node_positions[v2])
                if cached_line_conflict(l1, l2):
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


    @cached_environment_method("obstacles")
    def _violates_obstacles(self, line):
        for zone in EnvironmentConfig.FORBIDDEN_ZONES:
            poly=cached_obstacle_polygon(zone['center'], zone['radius'])
            if cached_line_conflict(line, poly):
                return True
        return False

    def _count_crossings(self, new_line, exclude_pair=None):
        count=0
        for u,v in self._get_unique_edges():
            if exclude_pair and {u,v}==set(exclude_pair): continue
            existing=cached_line(self.node_positions[u], self.node_positions[v])
            if cached_line_conflict(new_line, existing): count+=1
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
            line=cached_line(self.node_positions[u], self.node_positions[v])
            if self._violates_obstacles(line):
                self._update_node_features()
                return self._get_state(), -30.0, False, {'obstacle_rejection': True}
        prev_iso=self._count_isolated_turbines()
        crossings=0
        if act_type==1:
            line=cached_line(self.node_positions[u], self.node_positions[v])
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

        r_terminal=0.0
        if done:
            r_terminal = 500.0 if curr_iso==0 else -(curr_iso**2)*20.0
            if curr_iso==0 and self.cost_weight>0:
                ci=self.calculate_lifecycle_cost(); cost_m=ci['total']/1e6
                self.best_cost_window.append(cost_m)

        r_cross = crossings*ConstraintConfig.CROSSING_PENALTY  # [v3-4] −40/crossing
        r_step  = -1.0

        total=r_milestone+r_delta+r_cost+r_topo+r_terminal+r_cross+r_progress+r_control+r_step
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


    @cached_environment_method("metrics")
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


    @cached_environment_method("metrics")
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


    @cached_environment_method("metrics")
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
        n_connected=sum(1 for i in range(self.num_turbines) if i in connected)
        return {'total': c_inv+c_op+c_wake+c_fatigue+c_yaw_act+c_fatigue_route+c_robust+c_robust_route,
                'c_inv': c_inv, 'c_op': c_op, 'c_wake': c_wake,
                'c_fatigue': c_fatigue, 'c_yaw_act': c_yaw_act,
                'c_fatigue_route': c_fatigue_route,
                'c_robust': c_robust, 'c_robust_route': c_robust_route,
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

    @cached_environment_method("topology")
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

    @cached_environment_method("metrics")
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

    @cached_environment_method("topology")
    def _total_cable_length(self):
        length=0.0; rows,cols=np.where(self.adjacency_matrix>0)
        for r,c in zip(rows,cols):
            if r<c: length+=np.linalg.norm(self.node_positions[r]-self.node_positions[c])
        return length

    def _mst_cable_length(self):
        reference = self._mst_reference_length()
        return self._total_cable_length() if reference is None else reference

    @cached_environment_method("mst")
    def _mst_reference_length(self):
        G=nx.Graph()
        for i in range(self.num_nodes):
            for j in range(i+1,self.num_nodes):
                d=np.linalg.norm(self.node_positions[i]-self.node_positions[j])
                if d<=EnvironmentConfig.MAX_CABLE_LENGTH: G.add_edge(i,j,weight=d)
        if G.number_of_nodes()<self.num_nodes: return None
        try:
            mst=nx.minimum_spanning_tree(G)
            return sum(d['weight'] for _,_,d in mst.edges(data=True))
        except Exception: return None


    @cached_environment_method("metrics")
    def validate_topology(self):
        iso=self._count_isolated_turbines(); edges=self._get_unique_edges()
        obs_viol=0
        for u,v in edges:
            line=cached_line(self.node_positions[u], self.node_positions[v])
            for zone in EnvironmentConfig.FORBIDDEN_ZONES:
                poly=cached_obstacle_polygon(zone['center'], zone['radius'])
                if cached_line_conflict(line, poly): obs_viol+=1
        cross_viol=0
        for i,(u1,v1) in enumerate(edges):
            for u2,v2 in edges[i+1:]:
                if {u1,v1}&{u2,v2}: continue
                l1=cached_line(self.node_positions[u1], self.node_positions[v1])
                l2=cached_line(self.node_positions[u2], self.node_positions[v2])
                if cached_line_conflict(l1, l2): cross_viol+=1
        site_viol=count_site_violations(self.turbine_positions)
        # [v3-4] Valid requires iso=0, no obstacles, AND crossings ≤ MAX_CROSSING_FOR_BEST
        is_valid = (site_viol==0 and iso==0 and obs_viol==0 and cross_viol<=ConstraintConfig.MAX_CROSSING_FOR_BEST)
        return {'fully_connected': iso==0, 'isolated_count': iso,
                'site_boundary_violations': int(site_viol),
                'obstacle_violations': obs_viol, 'crossing_violations': cross_viol,
                'is_valid': is_valid}


# ============================================================================
# SECTION 8: NSGA-II LAYOUT OPTIMISER  [Innovation I2]
# ============================================================================

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
        self.final_robust_shortfall=0.0
        self.final_power_cvar=0.0
        self.final_power_std=0.0
        self.final_topology_feedback={}
        self._layout_topology_cache={}

    def _build_climate_scenarios(self):
        return build_wind_scenarios()

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
                objs.append(res[:5]); valid.append(res[5])
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
            # Elitist NSGA-II: keep the best solutions from parents + offspring.
            combined=population+offspring
            c_objs=[]; c_valid=[]
            for layout in combined:
                res=self._eval_objectives(layout)
                c_objs.append(res[:5]); c_valid.append(res[5])
            population=self._select_next_generation(combined, c_objs, c_valid, pop_size)

            if gen%10==0 and vi:
                best_idx=mfronts[0][0] if mfronts and mfronts[0] else vi[0]
                bo=objs[best_idx]
                LOG.info(f"  NSGA-II Gen {gen:3d} | valid={len(vi)}/{pop_size} | "
                         f"power={bo[0]:.2f} MW | wake={bo[1]:.3f} | "
                         f"load={bo[2]:.4f} | cost={bo[3]/1e6:.2f}M | "
                         f"robust={bo[4]:.3f} MW")
                nsga_log.append({'gen':gen,'valid':len(vi),'power_mw':round(bo[0],4),
                                 'wake_ratio':round(bo[1],5),'load_index':round(bo[2],5),
                                 'cost_m':round(bo[3]/1e6,4),
                                 'robust_shortfall_mw':round(bo[4],5)})

        save_json({'nsga2_progress': nsga_log}, "nsga2_convergence.json")
        final_objs=[]; final_valid=[]
        for layout in population:
            res=self._eval_objectives(layout)
            final_objs.append(res[:5]); final_valid.append(res[5])
        fvi=[i for i,f in enumerate(final_valid) if f]
        if fvi:
            fv_objs=[final_objs[i] for i in fvi]
            ffronts=self._fast_non_dom_sort(fv_objs)
            if ffronts and ffronts[0]:
                best_layout=population[fvi[ffronts[0][0]]]
            res=self._eval_objectives(best_layout)
            self.final_power=float(res[0]); self.final_wake_ratio=float(res[1])
            self.final_load_index=float(res[2]); self.cable_cost=float(res[3])
            self.final_robust_shortfall=float(res[4])
        return best_layout

    def _select_next_generation(self, population, objs, valid, pop_size):
        vi=[i for i,f in enumerate(valid) if f]
        if not vi:
            return [self._rand_feasible() for _ in range(pop_size)]

        v_objs=[objs[i] for i in vi]
        fronts=self._fast_non_dom_sort(v_objs)
        cd=self._crowding_distance(v_objs, fronts)
        selected=[]

        for front in fronts:
            if len(selected)+len(front)<=pop_size:
                selected.extend([vi[i] for i in front])
            else:
                remain=pop_size-len(selected)
                ranked=sorted(front, key=lambda i: cd[i], reverse=True)
                selected.extend([vi[i] for i in ranked[:remain]])
                break

        next_pop=[population[i].copy() for i in selected]
        while len(next_pop)<pop_size:
            next_pop.append(self._rand_feasible())
        return next_pop

    @cached_layout_objectives
    def _eval_objectives(self, layout):
        n=self.num_turbines; R=self.R
        feasible=self._is_layout_feasible(layout)
        if not feasible:
            return (-1e9, 1e3, 1e3, 1e12, 1e3, False)

        weighted_power=0.0; weighted_no_wake=0.0; weight_sum=0.0
        scenario_power=[]; scenario_weights=[]
        weighted_load=0.0

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
            weighted_power+=p_now*w
            weighted_no_wake+=p_no_wk*w
            weight_sum+=w
            scenario_power.append(float(p_now))
            scenario_weights.append(float(w))

            ti_arr=np.asarray(ti_s, dtype=float)
            ti_std=float(np.std(ti_arr))
            ti_norm=(np.maximum(ti_arr, 0.0) /
                     max(LifecycleConfig.FATIGUE_REFERENCE_TI, 1e-6))
            fatigue_damage=float(np.mean(
                ti_norm ** LifecycleConfig.FATIGUE_DAMAGE_EXPONENT))
            weighted_load+=(ti_std+
                            EnvironmentConfig.LAYOUT_FATIGUE_DAMAGE_WEIGHT*
                            fatigue_damage)*w

        weight_sum=max(weight_sum, 1e-9)
        expected_power_mw=weighted_power/weight_sum
        no_wake_mw=weighted_no_wake/weight_sum
        wake_ratio=max(0.0,(no_wake_mw-expected_power_mw)/(no_wake_mw+1e-6))
        load_idx=weighted_load/weight_sum

        p_arr=np.asarray(scenario_power, dtype=float)
        w_arr=np.asarray(scenario_weights, dtype=float)
        w_arr=w_arr/(np.sum(w_arr)+1e-12)
        power_std=float(np.sqrt(np.sum(w_arr*(p_arr-expected_power_mw)**2)))
        order=np.argsort(p_arr)
        alpha=max(min(EnvironmentConfig.LAYOUT_CVAR_ALPHA, 1.0), 1e-6)
        acc_w=0.0; acc_p=0.0
        for idx in order:
            take=min(float(w_arr[idx]), alpha-acc_w)
            if take<=0: break
            acc_p+=float(p_arr[idx])*take
            acc_w+=take
        cvar_power=acc_p/max(acc_w, 1e-9)
        robust_shortfall=max(0.0, expected_power_mw-cvar_power)
        robust_shortfall+=EnvironmentConfig.LAYOUT_POWER_STD_WEIGHT*power_std
        self.final_power_cvar=float(cvar_power)
        self.final_power_std=float(power_std)

        topo=self._fast_topology_feedback(layout)
        topology_cost=topo['total_cost']
        robust_shortfall+=EnvironmentConfig.LAYOUT_TOPOLOGY_RISK_MW_PENALTY*topo['n1_risk']
        self.final_topology_feedback=topo
        if not topo['feasible']:
            return (expected_power_mw, wake_ratio, load_idx, topology_cost,
                    robust_shortfall+EnvironmentConfig.LAYOUT_TOPOLOGY_RISK_MW_PENALTY,
                    False)

        return (expected_power_mw, wake_ratio, load_idx, topology_cost,
                robust_shortfall, True)

    def _fast_topology_feedback(self, layout):
        key=layout_cache_key(self, layout, globals())
        cached=self._layout_topology_cache.get(key) if caches_enabled() else None
        if cached is not None:
            return cached

        n=self.num_turbines; sub=n
        sub_pos=LARGE_CASE_SUBSTATION_POSITIONS[0].copy()
        full_pos=np.vstack([layout, sub_pos.reshape(1,2)])
        candidate=[]; G=nx.Graph(); G.add_nodes_from(range(n+1))

        for i in range(n+1):
            for j in range(i+1,n+1):
                d=float(np.linalg.norm(full_pos[i]-full_pos[j]))
                if d>EnvironmentConfig.MAX_CABLE_LENGTH:
                    continue
                line=LineString([full_pos[i], full_pos[j]])
                if self._line_hits_forbidden(line):
                    continue
                candidate.append((i,j,d,line))
                G.add_edge(i,j,weight=d)

        if G.number_of_edges()==0 or not nx.is_connected(G):
            out={'feasible':False,'total_cost':1e12,'cable_cost':1e12,
                 'cable_m':0.0,'crossings':0,'obstacle_violations':0,
                 'max_single_edge_outage_turbines':n,'n1_risk':1.0,
                 'redundancy_ratio':0.0,'edges':[]}
            self._cache_topology_feedback(key,out)
            return out

        mst=nx.minimum_spanning_tree(G, weight='weight')
        chosen={tuple(sorted((u,v))) for u,v in mst.edges()}
        deg=dict(mst.degree())
        extra_budget=max(1, int(math.ceil(EnvironmentConfig.LAYOUT_TOPOLOGY_EXTRA_EDGE_RATIO*n)))
        extras=0

        for i,j,d,line in sorted(candidate, key=lambda e:e[2]):
            if extras>=extra_budget:
                break
            e=tuple(sorted((i,j)))
            if e in chosen:
                continue
            need_i=(i<n and deg.get(i,0)<EnvironmentConfig.LAYOUT_TOPOLOGY_MIN_DEGREE)
            need_j=(j<n and deg.get(j,0)<EnvironmentConfig.LAYOUT_TOPOLOGY_MIN_DEGREE)
            if not (need_i or need_j):
                continue
            if self._line_crosses_chosen_edges(line, e, chosen, full_pos):
                continue
            chosen.add(e)
            deg[i]=deg.get(i,0)+1; deg[j]=deg.get(j,0)+1
            extras+=1

        cable_m=sum(float(np.linalg.norm(full_pos[u]-full_pos[v])) for u,v in chosen)
        cable_cost=(cable_m/1000.0)*LifecycleConfig.CABLE_COST_PER_KM
        crossings=self._edge_crossing_count(chosen, full_pos)
        max_outage,n1_risk=self._single_edge_outage_risk(chosen, n, sub)
        redundancy_ratio=max(0.0, (len(chosen)-n)/max(n,1))

        n1_penalty=n1_risk*LifecycleConfig.CABLE_COST_PER_KM*EnvironmentConfig.LAYOUT_N1_RISK_COST_KM
        cross_penalty=crossings*LifecycleConfig.CABLE_COST_PER_KM*EnvironmentConfig.LAYOUT_CROSSING_COST_KM
        total_cost=cable_cost+n1_penalty+cross_penalty

        out={'feasible':True,'total_cost':float(total_cost),'cable_cost':float(cable_cost),
             'cable_m':float(cable_m),'crossings':int(crossings),
             'obstacle_violations':0,'max_single_edge_outage_turbines':int(max_outage),
             'n1_risk':float(n1_risk),'redundancy_ratio':float(redundancy_ratio),
             'edges':[(int(u),int(v)) for u,v in sorted(chosen)]}
        self._cache_topology_feedback(key,out)
        return out

    def _cache_topology_feedback(self, key, value):
        if not caches_enabled(): return
        if len(self._layout_topology_cache)>5000:
            self._layout_topology_cache.clear()
        self._layout_topology_cache[key]=value

    def _line_hits_forbidden(self, line):
        for zone in EnvironmentConfig.FORBIDDEN_ZONES:
            poly=cached_obstacle_polygon(zone['center'], zone['radius'])
            if cached_line_conflict(line, poly):
                return True
        return False

    def _line_crosses_chosen_edges(self, line, edge, chosen, full_pos):
        u,v=edge
        for a,b in chosen:
            if {u,v} & {a,b}:
                continue
            other=LineString([full_pos[a], full_pos[b]])
            if cached_line_conflict(line, other):
                return True
        return False

    def _edge_crossing_count(self, chosen, full_pos):
        edges=list(chosen); count=0
        for idx,(u1,v1) in enumerate(edges):
            l1=LineString([full_pos[u1], full_pos[v1]])
            for u2,v2 in edges[idx+1:]:
                if {u1,v1} & {u2,v2}:
                    continue
                l2=LineString([full_pos[u2], full_pos[v2]])
                if cached_line_conflict(l1, l2):
                    count+=1
        return count

    def _single_edge_outage_risk(self, chosen, n, sub):
        H=nx.Graph(); H.add_nodes_from(range(n+1)); H.add_edges_from(chosen)
        if not nx.is_connected(H):
            return n,1.0
        max_outage=0
        for e in list(chosen):
            T=H.copy(); T.remove_edge(*e)
            if sub in T:
                comp=nx.node_connected_component(T, sub)
                outage=sum(1 for i in range(n) if i not in comp)
            else:
                outage=n
            max_outage=max(max_outage, outage)
        return max_outage, max_outage/max(n,1)

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
        min_spacing = (EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR
                       * EnvironmentConfig.ROTOR_DIAMETER)
        return layout_violation_score(
            layout,
            min_spacing=min_spacing,
            forbidden_zones=EnvironmentConfig.FORBIDDEN_ZONES,
            obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN,
        )

    def _is_layout_feasible(self, layout, tol=1e-6):
        return self._layout_violation_score(layout)<=tol

    def _project_feasible(self, layout, iters=None):
        repair_iterations = 120 if iters is None else max(1, int(iters))
        return repair_large_case_layout(
            layout,
            n_turbines=self.num_turbines,
            min_spacing=(EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR
                         * EnvironmentConfig.ROTOR_DIAMETER),
            forbidden_zones=EnvironmentConfig.FORBIDDEN_ZONES,
            obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN,
            rng=np.random,
            iterations=repair_iterations,
        )

    def _rand_feasible(self):
        return sample_large_case_layout(
            n_turbines=self.num_turbines,
            min_spacing=(EnvironmentConfig.MIN_TURBINE_DISTANCE_FACTOR
                         * EnvironmentConfig.ROTOR_DIAMETER),
            forbidden_zones=EnvironmentConfig.FORBIDDEN_ZONES,
            obstacle_margin=ConstraintConfig.OBSTACLE_HARD_MARGIN,
            rng=np.random,
        )


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
                if any(cached_line_conflict(line, p) for p in self.forbidden_polys):
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
        self.episode_log=[]

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
                    conn_hist.append(iso_cnt)

                    #  Best topology requires crossings ≤ MAX_CROSSING_FOR_BEST
                    if val_check['is_valid'] and cost_m<self.min_cost:
                        self.min_cost=cost_m
                        self.best_topo={'adj_matrix':self.env.adjacency_matrix.copy(),
                                        'edge_index':self.env.edge_index.clone(),
                                        'edge_attr' :self.env.edge_attr.clone(),
                                        'yaw_angles':self.env.yaw_angles.copy(),
                                        'cost_m':cost_m,'episode':ep_cnt,'validation':val_check}
                        LOG.info(f"[BEST TOPOLOGY] ep={ep_cnt} cost={cost_m:.3f}M | "
                                 f"iso={iso_cnt} | obs={val_check['obstacle_violations']} | "
                                 f"cross={val_check['crossing_violations']}")

                    #  Secondary tracker: best cost ignoring crossing constraint
                    if (iso_cnt==0 and val_check['obstacle_violations']==0
                            and cost_m<self.min_cost_unrestricted):
                        self.min_cost_unrestricted=cost_m
                        self.best_cost_topo={'adj_matrix':self.env.adjacency_matrix.copy(),
                                             'edge_index':self.env.edge_index.clone(),
                                             'edge_attr' :self.env.edge_attr.clone(),
                                             'yaw_angles':self.env.yaw_angles.copy(),
                                             'cost_m':cost_m,'episode':ep_cnt,
                                             'crossing_violations':val_check['crossing_violations']}
                        LOG.info(f"[BEST COST (unrestricted)] ep={ep_cnt} cost={cost_m:.3f}M | "
                                 f"cross={val_check['crossing_violations']}")

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
                        'c_fatigue_route_m':round(ci.get('c_fatigue_route',0.0)/1e6,4),
                        'c_yaw_act_m':round(ci.get('c_yaw_act',0.0)/1e6,4),
                        'c_robust_m':round(ci.get('c_robust',0.0)/1e6,4),
                        'c_robust_route_m':round(ci.get('c_robust_route',0.0)/1e6,4),
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

        return rewards_hist, self.best_topo, self.best_cost_topo


# ============================================================================
# SECTION 15b: POST-PPO GREEDY EDGE-SWAP REFINEMENT
# ============================================================================

def greedy_edge_swap_refinement(env, max_iters=200):
    """
     Deterministic edge-swap local search to reduce cable cost after PPO.
    At each iteration, tries to replace the longest cable with a shorter alternative
    that maintains full connectivity and does not create new obstacle violations.
    Operates on env's current topology in-place.
    Returns: dict with improvement stats.
    """
    LOG.info("  [v6-5] Running post-PPO greedy edge-swap refinement ...")
    initial_cost = env.calculate_lifecycle_cost()['total'] / 1e6
    initial_len  = env._total_cable_length() / 1000.0
    improved = 0

    for it in range(max_iters):
        edges = env._get_unique_edges()
        if not edges: break
        # Find longest edge
        edge_lens = [(np.linalg.norm(env.node_positions[u] - env.node_positions[v]), u, v)
                     for u, v in edges]
        edge_lens.sort(reverse=True)
        swapped = False
        for _, u, v in edge_lens:
            # Try removing this edge and adding a shorter alternative path
            # Only proceed if removing doesn't disconnect the graph
            env.adjacency_matrix[u, v] = env.adjacency_matrix[v, u] = 0
            if env._count_isolated_turbines() > 0:
                # Removing disconnects – try to find a better reconnection
                # Enumerate alternative edges (shorter than current) connecting same pair's component
                best_alt = None; best_alt_len = np.linalg.norm(env.node_positions[u]-env.node_positions[v])
                for i in range(env.num_nodes):
                    for j in range(i+1, env.num_nodes):
                        if env.adjacency_matrix[i, j] > 0: continue
                        d = np.linalg.norm(env.node_positions[i] - env.node_positions[j])
                        if d >= best_alt_len: continue
                        line = LineString([env.node_positions[i], env.node_positions[j]])
                        if env._violates_obstacles(line): continue
                        # Temporarily add and check connectivity
                        env.adjacency_matrix[i, j] = env.adjacency_matrix[j, i] = 1
                        if env._count_isolated_turbines() == 0:
                            cross = env._count_crossings(line, exclude_pair=(i, j))
                            if cross <= ConstraintConfig.MAX_CROSSING_FOR_BEST:
                                best_alt = (i, j); best_alt_len = d
                        env.adjacency_matrix[i, j] = env.adjacency_matrix[j, i] = 0
                if best_alt is not None:
                    i2, j2 = best_alt
                    env._add_edge(i2, j2)
                    env._update_node_features()
                    swapped = True; improved += 1; break
                else:
                    # Restore original edge
                    env.adjacency_matrix[u, v] = env.adjacency_matrix[v, u] = 1
            else:
                # Removing didn't disconnect – keep it removed (shorter topology)
                # Rebuild edge structures
                env._remove_edge(u, v)   # already removed from adj, rebuild tensors
                env.adjacency_matrix[u, v] = env.adjacency_matrix[v, u] = 0
                env._update_node_features()
                swapped = True; improved += 1; break
        if not swapped:
            break

    final_cost = env.calculate_lifecycle_cost()['total'] / 1e6
    final_len  = env._total_cable_length() / 1000.0
    delta_cost = initial_cost - final_cost
    LOG.info(f"  [v6-5] Refinement done | iters_improved={improved} | "
             f"cost {initial_cost:.3f}→{final_cost:.3f} M RMB (Δ={delta_cost:+.3f}) | "
             f"cable {initial_len:.2f}→{final_len:.2f} km")
    return {'initial_cost_m': round(initial_cost, 4), 'final_cost_m': round(final_cost, 4),
            'delta_cost_m': round(delta_cost, 4), 'iters_improved': improved,
            'initial_cable_km': round(initial_len, 4), 'final_cable_km': round(final_len, 4)}


def feasibility_first_topology_repair(env):
    """Replace an invalid graph with a connected, obstacle-safe low-crossing tree."""
    initial_validation = env.validate_topology()
    if initial_validation.get('is_valid', False):
        return {
            'repaired': False,
            'method': 'not-required',
            'initial_edges': len(env._get_unique_edges()),
            'final_edges': len(env._get_unique_edges()),
            'initial_crossings': initial_validation.get('crossing_violations', 0),
            'final_crossings': initial_validation.get('crossing_violations', 0),
            'is_valid': True,
        }

    LOG.warning(
        "  [repair] Invalid topology before objective search: edges=%d, isolated=%d, "
        "obstacles=%d, crossings=%d. Rebuilding a feasibility-first tree.",
        len(env._get_unique_edges()),
        initial_validation.get('isolated_count', 0),
        initial_validation.get('obstacle_violations', 0),
        initial_validation.get('crossing_violations', 0),
    )
    original = (env.adjacency_matrix.copy(), env.edge_index.clone(), env.edge_attr.clone())
    original_edges = {tuple(sorted(edge)) for edge in env._get_unique_edges()}
    candidates = []
    for i in range(env.num_nodes):
        for j in range(i + 1, env.num_nodes):
            dist = float(np.linalg.norm(env.node_positions[i] - env.node_positions[j]))
            if dist > EnvironmentConfig.MAX_CABLE_LENGTH + 1e-9:
                continue
            line = LineString([env.node_positions[i], env.node_positions[j]])
            if env._violates_obstacles(line):
                continue
            candidates.append((dist, i, j, line))
    candidates.sort(key=lambda item: (
        item[0], 0 if (item[1], item[2]) in original_edges else 1
    ))

    selected = []
    selected_pairs = set()
    components = nx.utils.UnionFind(range(env.num_nodes))

    def crossing_count(line, i, j):
        count = 0
        for _, u, v, existing in selected:
            if {i, j} & {u, v}:
                continue
            if cached_line_conflict(line, existing):
                count += 1
        return count

    # Kruskal on geometric edges normally yields a crossing-free Euclidean tree.
    for dist, i, j, line in candidates:
        if components[i] == components[j] or crossing_count(line, i, j) > 0:
            continue
        selected.append((dist, i, j, line))
        selected_pairs.add((i, j))
        components.union(i, j)
        if len(selected) == env.num_nodes - 1:
            break

    # Obstacle or cable-length constraints can block a planar Kruskal pass.
    # Connect any remaining components with the minimum additional crossings.
    while len(selected) < env.num_nodes - 1:
        best = None
        for dist, i, j, line in candidates:
            if (i, j) in selected_pairs or components[i] == components[j]:
                continue
            key = (crossing_count(line, i, j), dist,
                   0 if (i, j) in original_edges else 1)
            if best is None or key < best[0]:
                best = (key, dist, i, j, line)
        if best is None:
            break
        _, dist, i, j, line = best
        selected.append((dist, i, j, line))
        selected_pairs.add((i, j))
        components.union(i, j)

    if len(selected) != env.num_nodes - 1:
        env.adjacency_matrix, env.edge_index, env.edge_attr = original
        env._update_node_features()
        raise RuntimeError(
            "Feasibility repair failed: legal cable graph cannot connect all nodes."
        )

    env.adjacency_matrix = np.zeros_like(env.adjacency_matrix)
    env.edge_index = torch.empty((2, 0), dtype=torch.long)
    env.edge_attr = torch.empty(
        (0, GNNConfig.EDGE_FEATURE_DIM - 1), dtype=torch.float32
    )
    for _, i, j, _ in selected:
        env._add_edge(i, j)
    env._update_node_features()
    final_validation = env.validate_topology()
    if not final_validation.get('is_valid', False):
        env.adjacency_matrix, env.edge_index, env.edge_attr = original
        env._update_node_features()
        raise RuntimeError(
            "Feasibility repair produced an invalid topology: "
            f"{final_validation}"
        )

    LOG.info(
        "  [repair] Feasible tree built: edges %d->%d, crossings %d->%d.",
        len(original_edges), len(selected),
        initial_validation.get('crossing_violations', 0),
        final_validation.get('crossing_violations', 0),
    )
    return {
        'repaired': True,
        'method': 'crossing-aware-spanning-tree',
        'initial_edges': len(original_edges),
        'final_edges': len(selected),
        'initial_crossings': initial_validation.get('crossing_violations', 0),
        'final_crossings': final_validation.get('crossing_violations', 0),
        'is_valid': True,
    }


def objective_edge_swap_refinement(env, max_iters=55, candidate_limit=90):
    """
    Re-optimised local search for v9: minimise the robust lifecycle objective,
    including uncertainty-aware routing penalties, rather than cable length only.
    """
    LOG.info("  [reopt] Running robust-objective edge-swap refinement ...")
    pre_repair_obj=env.calculate_lifecycle_cost()['total']/1e6
    pre_repair_len=env._total_cable_length()/1000.0
    repair_stats=feasibility_first_topology_repair(env)
    initial_obj=env.calculate_lifecycle_cost()['total']/1e6
    initial_len=env._total_cable_length()/1000.0
    improved=0

    def snapshot():
        return (env.adjacency_matrix.copy(), env.edge_index.clone(), env.edge_attr.clone())

    def restore(snap):
        env.adjacency_matrix, env.edge_index, env.edge_attr=snap
        env._update_node_features()

    def feasible():
        val=env.validate_topology()
        return val.get('is_valid', False)

    best_obj=initial_obj
    for _ in range(max_iters):
        edges=env._get_unique_edges()
        if not edges: break
        edge_order=sorted(
            [(np.linalg.norm(env.node_positions[u]-env.node_positions[v]),u,v) for u,v in edges],
            reverse=True)
        accepted=False
        for _,u,v in edge_order:
            base=snapshot()
            env._remove_edge(u,v)
            candidates=[]
            for i in range(env.num_nodes):
                for j in range(i+1, env.num_nodes):
                    if {i,j}=={u,v}: continue
                    if env.adjacency_matrix[i,j]>0: continue
                    line=LineString([env.node_positions[i],env.node_positions[j]])
                    if env._violates_obstacles(line): continue
                    dist=np.linalg.norm(env.node_positions[i]-env.node_positions[j])
                    if dist>EnvironmentConfig.MAX_CABLE_LENGTH+1e-9: continue
                    cross=env._count_crossings(line, exclude_pair=(i,j))
                    candidates.append((cross, dist, i, j))
            candidates.sort(key=lambda x:(x[0], x[1]))
            if candidate_limit:
                candidates=candidates[:candidate_limit]
            best_pair=None; best_pair_obj=best_obj
            for _,_,i,j in candidates:
                env._add_edge(i,j)
                if feasible():
                    obj=env.calculate_lifecycle_cost()['total']/1e6
                    if obj<best_pair_obj-1e-5:
                        best_pair_obj=obj; best_pair=(i,j)
                env._remove_edge(i,j)
            if best_pair is not None:
                env._add_edge(best_pair[0], best_pair[1])
                env._update_node_features()
                best_obj=best_pair_obj
                improved+=1; accepted=True
                break
            restore(base)
        if not accepted:
            break

    final_obj=env.calculate_lifecycle_cost()['total']/1e6
    final_len=env._total_cable_length()/1000.0
    LOG.info(f"  [reopt] Refinement done | iters_improved={improved} | "
             f"objective {initial_obj:.3f}->{final_obj:.3f} M RMB | "
             f"total {pre_repair_obj:.3f}->{final_obj:.3f} M RMB | "
             f"cable {pre_repair_len:.2f}->{final_len:.2f} km")
    return {'initial_cost_m':round(pre_repair_obj,4),'post_repair_cost_m':round(initial_obj,4),
            'final_cost_m':round(final_obj,4),
            'delta_cost_m':round(pre_repair_obj-final_obj,4),
            'objective_delta_cost_m':round(initial_obj-final_obj,4),
            'iters_improved':improved,
            'initial_cable_km':round(pre_repair_len,4),
            'post_repair_cable_km':round(initial_len,4),'final_cable_km':round(final_len,4),
            'feasibility_repair':repair_stats,
            'objective':'robust-aware-lifecycle'}


# ============================================================================
# SECTION 16: RESULTS EXPORT
# ============================================================================

def export_results_json(comparison_jensen, comparison_cfd, layout_metrics,
                        pretrain_stats, episode_log, best_topo_info,
                        final_validation, final_cost, env, stage_log,
                        refinement_stats=None):
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
    save_json({'description':'Final topology graph','case_definition':case_metadata(),
               'nodes':node_data,'edges':edges_data,
               'validation':final_validation,'cost':final_cost}, "fig8_final_topology.json")
    save_json({'description':'Paper summary metrics','case_definition':case_metadata(),
               'best_topology':best_topo_info,
               'layout':layout_metrics,'pretrain':pretrain_stats,
               'n_episodes':len(episode_log),'stage_transitions':stage_log,
               'refinement':refinement_stats or {}},
              "paper_summary_table.json")


# ============================================================================
# SECTION 17: MAIN PIPELINE
# ============================================================================

@cache_reported_main
def main():
    t_global=time.perf_counter()
    device=torch.device(DeviceConfig.DEVICE)

    log_section("CFD-GNN-PPO Robust Tri-Level Fatigue Framework  v9.0")
    LOG.info(f"Device={device}  Seed={RANDOM_SEED}")
    LOG.info(f"Large case={CASE_NAME} | case_id={CASE_ID}")
    LOG.info(f"Grid={EnvironmentConfig.GRID_SIZE} m  Turbines={EnvironmentConfig.NUM_TURBINES}  "
             f"Rated={EnvironmentConfig.RATED_POWER} MW  D={EnvironmentConfig.ROTOR_DIAMETER} m (IEA 10 MW reference turbine)")
    LOG.info(f"v9 key changes: wind-speed/direction uncertainty, yaw tracking error, "
             f"CVaR power shortfall, robust TI penalties, and robust-route topology selection")

    n_turbines=EnvironmentConfig.NUM_TURBINES
    domain_bounds={'x':(0,EnvironmentConfig.GRID_SIZE[0]),'y':(0,EnvironmentConfig.GRID_SIZE[1])}

    results_jensen, results_cfd, best_layout_cfd, avg_imp = \
        run_baseline_comparison(n_turbines, domain_bounds)

    env=OffshoreWindEnv(initial_layout=best_layout_cfd)
    nsga_optimizer=HybridAICFDLayoutOptimizer(env)
    best_layout=nsga_optimizer.optimize_layout()

    env=OffshoreWindEnv(initial_layout=best_layout)
    layout_metrics={
        'power_mw'       : round(nsga_optimizer.final_power,4),
        'wake_loss_ratio': round(nsga_optimizer.final_wake_ratio,5),
        'load_index'     : round(nsga_optimizer.final_load_index,5),
        'cable_cost_m'   : round(nsga_optimizer.cable_cost/1e6,4),
        'topology_feedback_cost_m': round(nsga_optimizer.cable_cost/1e6,4),
        'layout_power_cvar_mw': round(nsga_optimizer.final_power_cvar,4),
        'layout_power_std_mw' : round(nsga_optimizer.final_power_std,4),
        'layout_robust_shortfall_mw': round(nsga_optimizer.final_robust_shortfall,4),
        'coopt_topology': nsga_optimizer.final_topology_feedback,
    }
    LOG.info(f"  NSGA-II | power={layout_metrics['power_mw']:.3f} MW | "
             f"wake_loss={layout_metrics['wake_loss_ratio']:.4f} | "
             f"TI_load_idx={layout_metrics['load_index']:.5f}")
    topo_m=layout_metrics.get('coopt_topology',{})
    LOG.info(f"  Co-optimised layout-topology feedback | "
             f"topo_cost={layout_metrics['topology_feedback_cost_m']:.3f} M RMB | "
             f"red={topo_m.get('redundancy_ratio',0.0):.3f} | "
             f"N1={topo_m.get('max_single_edge_outage_turbines',0)} | "
             f"cross={topo_m.get('crossings',0)}")

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
    rewards_hist, best_topo, best_cost_topo=trainer.train(TrainingConfig.NUM_EPISODES)

    log_section("Phase 4 – Final Evaluation & JSON Export")

    best_topo_info={}; final_validation={}; final_cost={}; refinement_stats={}; final_control={}

    active_topo = best_topo
    if active_topo is None and best_cost_topo is not None:
        LOG.warning("  No crossing-compliant topology found; using best-cost topology "
                    f"(cross={best_cost_topo['crossing_violations']})")
        active_topo = best_cost_topo
    elif (active_topo is not None and best_cost_topo is not None
          and best_cost_topo['cost_m'] < active_topo['cost_m'] - 0.01):
        LOG.info(f"  [v6-3] best_cost_topo ({best_cost_topo['cost_m']:.3f}M) is cheaper than "
                 f"best_valid_topo ({active_topo['cost_m']:.3f}M); both reported.")

    if active_topo is not None:
        LOG.info(f"  Restoring best topology (ep={active_topo['episode']}, "
                 f"cost={active_topo['cost_m']:.3f}M)")
        env.adjacency_matrix=active_topo['adj_matrix']
        env.edge_index=active_topo['edge_index']
        env.edge_attr =active_topo['edge_attr']
        env.yaw_angles=np.asarray(active_topo.get('yaw_angles', np.zeros(env.num_turbines)), dtype=float)
        env.wake_cache['is_valid']=False
        env._update_node_features()

        # Re-optimised v9: refine with the robust-aware lifecycle objective,
        # not with cable length alone.
        refinement_stats = objective_edge_swap_refinement(env, max_iters=55)

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
        LOG.info(f"    C_fatigue_route   : {final_cost.get('c_fatigue_route',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_yaw_act         : {final_cost.get('c_yaw_act',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_robust          : {final_cost.get('c_robust',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_robust_route    : {final_cost.get('c_robust_route',0.0)/1e6:.3f} M RMB")
        LOG.info(f"    C_total           : {final_cost['total']/1e6:.3f} M RMB")
        LOG.info("  POWER STATISTICS")
        LOG.info(f"    Deliverable Power : {final_cost['deliverable_power']:.4f} MW")
        LOG.info(f"    Ohmic Loss        : {final_cost['ohmic_loss']:.6f} MW")
        LOG.info(f"    Net Power         : {final_cost['net_power']:.4f} MW")
        LOG.info(f"    CVaR Power        : {final_cost.get('cvar_power_mw',final_cost['net_power']):.4f} MW")
        LOG.info(f"    Worst Power       : {final_cost.get('worst_power_mw',final_cost['net_power']):.4f} MW")
        LOG.info(f"    Connected Turbines: {final_cost['connected_turbines']}/{env.num_turbines}")
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
            'yaw_control'  : {k:round(float(v),6) for k,v in final_control.items()},
            'robust'       : {k:round(float(final_cost.get(k,0.0)),6)
                              for k in ['c_robust','c_robust_power','c_robust_ti',
                                        'c_robust_route',
                                        'expected_power_mw','cvar_power_mw','worst_power_mw',
                                        'expected_max_ti','worst_max_ti']},
        }
        if best_cost_topo is not None:
            best_topo_info['best_cost_unrestricted_m'] = round(best_cost_topo['cost_m'],4)
            best_topo_info['best_cost_unrestricted_ep'] = best_cost_topo['episode']

        ckpt_path=os.path.join(CHECKPOINT_DIR,"best_actor_critic_v9_robust.pt")
        torch.save({'actor_critic_state':actor_critic.state_dict(),
                    'best_episode':active_topo['episode'],'cost_m':active_topo['cost_m'],
                    'yaw_angles':env.yaw_angles,
                    'version':'v9.0','innovations':['I1-Frandsen','I2-NSGA-II','I3-GATv2',
                                                    'I4-PPO-IQN','I5-Lifecycle','I6-Benchmark',
                                                    'I7-Active-Yaw-Wake-Steering',
                                                    'I8-Fatigue-Economics',
                                                    'I9-Robust-Uncertainty'],
                    'v9_changes':['wind-uncertainty-scenarios','yaw-tracking-error',
                                  'cvar-power-shortfall','robust-ti-penalty']},
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
                   'Metadata':{'Case_Definition':case_metadata(),'Algorithm':'CFD-GNN-PPO Robust Tri-Level Fatigue v9.0',
                                'Total_Power_MW':round(float(final_cost['net_power']),4),
                                'N_Turbines':env.num_turbines,'Version':'v9.0',
                                'Key_Changes':['Frandsen-TI-quadrature-v5','18-dim-yaw-features',
                                               'cosine-LR','crossing-threshold-5',
                                               'unified-turbine-params',
                                               'entropy-PID','entropy-refresh-v6',
                                               'dynamic-cost-reward','greedy-edge-swap-v6',
                                               'secondary-best-tracker-v6',
                                               'active-yaw-wake-steering-v7',
                                               '18-dim-yaw-state',
                                               'hybrid-discrete-continuous-PPO',
                                               'fatigue-economics-v8',
                                               'robust-uncertainty-v9']}},
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
        refinement_stats=refinement_stats)

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
