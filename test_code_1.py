import numpy as np


# ============================================================
#  UNIT CONVERSIONS
# ============================================================

def N_to_grams_force(Tn):
    """Convert Newtons -> grams-force."""
    return (Tn / 9.81) * 1000.0


def grams_force_to_N(Tg):
    """Convert grams-force -> Newtons."""
    return Tg / 1000.0 * 9.81


# ============================================================
#  RPM -> THRUST (N)
#  From Bitcraze polynomial: thrust in grams-force
# ============================================================

def rpm_to_thrust_N(rpm):
    """
    Convert motor RPM -> thrust in Newtons using Bitcraze empirical model.
    
    thrust_gf = 1.0942e-07 * rpm^2 - 2.1059e-04 * rpm + 0.154
    """
    rpm = np.asarray(rpm)

    # thrust in grams-force
    Tg = 1.0942e-07 * rpm**2 - 2.1059e-04 * rpm + 0.154
    Tg = np.maximum(Tg, 0)  # no negative thrust

    return grams_force_to_N(Tg)


# ============================================================
#  THRUST (N) -> RPM (solve quadratic equation)
# ============================================================

def thrust_N_to_rpm(Tn):
    """
    Solve for RPM needed to generate thrust Tn (Newtons).
    Uses quadratic formula on Bitcraze thrust model.
    """

    # Convert desired thrust to grams-force
    Tg = N_to_grams_force(Tn)

    # Polynomial: a*rpm^2 + b*rpm + c = Tg
    a = 1.0942e-07
    b = -2.1059e-04
    c = 0.154 - Tg

    disc = b**2 - 4 * a * c
    if disc < 0:
        raise ValueError("Thrust too high: no real RPM solution.")

    rpm1 = (-b + np.sqrt(disc)) / (2 * a)
    rpm2 = (-b - np.sqrt(disc)) / (2 * a)

    # Use the positive, physical root
    return max(rpm1, rpm2)


# ============================================================
#  PWM -> THRUST (N)
#  Using measured PWM vs thrust (grams-force) quadratic fit
# ============================================================

def pwm_to_thrust_N(pwm, pwm_max=65535):
    """
    Convert PWM command -> thrust in Newtons.
    Fit derived from Bitcraze measured data (total thrust of quad):

    Tg = a*p^2 + b*p + c
    p = PWM percentage (0..100)
    """
    pwm = np.asarray(pwm)
    pct = (pwm / pwm_max) * 100.0

    # Fit coefficients (grams-force)
    a = 0.0026515
    b = 0.3622513
    c = -0.1140069

    Tg = a * pct**2 + b * pct + c
    Tg = np.maximum(Tg, 0)

    return grams_force_to_N(Tg)


# ============================================================
#  THRUST (N) -> PWM
# ============================================================

def thrust_N_to_pwm(Tn, pwm_max=65535):
    """
    Invert PWM->thrust curve.
    Tg = a*p^2 + b*p + c
    Solve for PWM% p, then convert to actual PWM count.
    """

    # Convert Newtons -> grams-force
    Tg = N_to_grams_force(Tn)

    # quadratic fit (grams-force)
    a = 0.0026515
    b = 0.3622513
    c = -0.1140069 - Tg

    disc = b**2 - 4 * a * c
    if disc < 0:
        raise ValueError("Thrust outside valid PWM model")

    p1 = (-b + np.sqrt(disc)) / (2 * a)
    p2 = (-b - np.sqrt(disc)) / (2 * a)

    # pick physically valid percentage
    p = max(p1, p2)
    p = np.clip(p, 0, 100)

    # Convert % → raw PWM count
    return (p / 100.0) * pwm_max


# ============================================================
# FULL PIPELINE CLASS
# ============================================================

class ThrustPipeline:
    """
    Convenience wrapper to use the whole thrust/RPM/PWM conversion pipeline.
    """

    def __init__(self, pwm_max=65535):
        self.pwm_max = pwm_max

    # ---- THRUST <-> RPM ----
    def thrust_to_rpm(self, Tn):
        return thrust_N_to_rpm(Tn)

    def rpm_to_thrust(self, rpm):
        return rpm_to_thrust_N(rpm)

    # ---- THRUST <-> PWM ----
    def thrust_to_pwm(self, Tn):
        return thrust_N_to_pwm(Tn, pwm_max=self.pwm_max)

    def pwm_to_thrust(self, pwm):
        return pwm_to_thrust_N(pwm, pwm_max=self.pwm_max)

    # ---- PWM <-> RPM (via thrust) ----
    def pwm_to_rpm(self, pwm):
        Tn = self.pwm_to_thrust(pwm)
        return self.thrust_to_rpm(Tn)

    def rpm_to_pwm(self, rpm):
        Tn = self.rpm_to_thrust(rpm)
        return self.thrust_to_pwm(Tn)


# ============================================================
# DEMO USAGE
# ============================================================

if __name__ == "__main__":
    pipe = ThrustPipeline(pwm_max=60000)  # Phoenix-sim style

    m = 0.027  # 27 grams
    T_hover = m * 9.81   # Newtons

    print("\n=== Crazyflie Hover Thrust ===")
    print("Hover thrust (N):", T_hover)

    rpm_hover = pipe.thrust_to_rpm(T_hover)
    pwm_hover = pipe.thrust_to_pwm(T_hover)

    print("RPM needed:", rpm_hover)
    print("PWM needed:", pwm_hover)

    # And reverse check:
    print("Check thrust from PWM:", pipe.pwm_to_thrust(pwm_hover))
    print("Check thrust from RPM:", pipe.rpm_to_thrust(rpm_hover))
