"""Physics-inspired synthetic accelerometer generators for the ward-event classes.

Walking (2) and Fall (3) come from the real SisFall recordings; every other
class has no public dataset, so we synthesise 1.5 s windows (75 samples at
50 Hz, units of g). process_ward_data.py uses these to build the training and
test sets, and sanity_check.py uses the very same functions to probe a trained
checkpoint, so the probe and the training distribution can never drift apart.

Axis convention follows SisFall's waist-mounted sensor: gravity reads about
-1 g on the Y axis when upright and about +1 g on Z when lying on the back.

Each generator fills a zero-initialised (WINDOW_SIZE, 3) array in place.
"""
import numpy as np

WINDOW_SIZE = 75


def f_lying(a):
    a[:, 2] = 1.0
    a += np.random.normal(0, np.random.uniform(0.01, 0.05), a.shape)  # Variable noise


def f_sit(a):
    a[:, 1] = 1.0
    a += np.random.normal(0, np.random.uniform(0.02, 0.06), a.shape)


def f_seiz(a):
    a[:, 2] = 1.0
    # Random frequency noise (some seizures are faster/slower)
    a += np.random.normal(0, np.random.uniform(0.3, 0.6), a.shape)


def f_slump(a):
    angle = np.random.uniform(0.6, 0.8)  # Variable slump angle
    a[:, 1] = angle
    a[:, 2] = angle
    a += np.random.normal(0, 0.02, a.shape)


def f_agit(a):
    freq = np.random.uniform(0.5, 2.0)  # Variable rolling speed
    t = np.linspace(0, freq * np.pi, WINDOW_SIZE)
    roll = np.sin(t) * np.random.uniform(0.4, 0.8)
    a[:, 2] = 1.0 - np.abs(roll)
    a[:, 0] = roll
    a += np.random.normal(0, 0.15, a.shape)


def f_choke(a):
    a[:, 1] = 0.9
    interval = np.random.randint(15, 30)  # Random cough spacing
    for k in range(5, 70, interval):
        force = np.random.uniform(1.2, 1.8)
        a[k:k + 3, 2] += force
        a[k:k + 3, 1] -= 0.5
    a += np.random.normal(0, 0.05, a.shape)


def f_vomit(a):
    a[:, 1] = 0.7
    a[:, 2] = 0.7
    freq = np.random.uniform(2.5, 3.5)  # Variable heave speed
    t = np.linspace(0, freq * np.pi, WINDOW_SIZE)
    h = np.sin(t) * np.random.uniform(0.6, 0.9)
    a[:, 2] += h
    a[:, 1] -= h * 0.3
    a += np.random.normal(0, 0.1, a.shape)


def f_cpr(a):
    a[:, 2] = 1.0
    freq = np.random.uniform(1.5, 2.2)  # Variable CPR speed (90-130 BPM)
    t = np.linspace(0, 1.5, WINDOW_SIZE)
    c = np.abs(np.sin(2 * np.pi * freq * t)) * np.random.uniform(0.7, 1.0)
    a[:, 2] += c
    a += np.random.normal(0, 0.05, a.shape)


def f_resp(a):
    a[:, 1] = 0.8
    freq = np.random.uniform(0.5, 1.2)  # Variable breathing rate
    t = np.linspace(0, 1.5, WINDOW_SIZE)
    b = np.sin(2 * np.pi * freq * t) * np.random.uniform(0.1, 0.2)
    a[:, 2] += b
    a += np.random.normal(0, 0.02, a.shape)


def f_trans(a):
    a[:, 2] = 1.0
    # Variable floor texture (rumble frequency)
    a += np.random.normal(0, np.random.uniform(0.05, 0.12), a.shape)


# --- Probes that imitate the *real* SisFall classes (not used for training) ---

def probe_walking(a):
    """Upright gait: -1 g gravity on Y, ~2 Hz stride oscillation plus heel strikes."""
    t = np.linspace(0, 1.5, WINDOW_SIZE)
    a[:, 1] = -1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    a[:, 2] = -0.2 + 0.3 * np.sin(2 * np.pi * 2.0 * t + 1.0)
    for i in range(5, WINDOW_SIZE, 25):  # 2 heel strikes per second
        a[i:i + 3, 1] -= 1.0
    a += np.random.normal(0, 0.25, a.shape)


def probe_fall(a):
    """Upright, one large impact spike, then lying still on the back."""
    a[:, 1] = -1.0
    a[35:40, 1] = 3.5
    a[35:40, 2] = 2.0
    a[45:, 1] = 0.0
    a[45:, 2] = 1.0
    a += np.random.normal(0, 0.05, a.shape)


# Index = class id (see ward_model.CLASSES). None = real-data class.
TRAINING_GENERATORS = [f_lying, f_sit, None, None, f_seiz, f_slump,
                       f_agit, f_choke, f_vomit, f_cpr, f_resp, f_trans]

# Generators for every class, for probing / demos.
GENERATORS = list(TRAINING_GENERATORS)
GENERATORS[2] = probe_walking
GENERATORS[3] = probe_fall


def make_window(class_id):
    """Return one fresh synthetic (WINDOW_SIZE, 3) window for the given class id."""
    a = np.zeros((WINDOW_SIZE, 3))
    GENERATORS[class_id](a)
    return a
