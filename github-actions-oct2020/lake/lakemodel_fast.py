import numpy as np
from scipy.optimize import root
from numba import jit
from numba.typed import List


@jit(nopython=True)
def calc_internal(seed, Pcrit, vars, b=0.42, q=2.0, mu=0.03, sigma=np.sqrt(10**(-5)),
                  delta=0.98, lake0=0, nobjs=4, nvars=6, nYears=100, nSamples=100, n=2, alpha=0.4,
                  inertia_threshold=-0.02, reliability_threshold=0.85):

    # Initialize arrays to store objective function values and constraints
    objs = [0.0] * nobjs

    # Set inflow distribution parameters
    log_sigma = np.sqrt(np.log(1 + sigma**2 / mu**2))
    log_mu = np.log(mu) - 0.5 * (log_sigma**2)

    # Initialize arrays to store average daily P and performance of solution in each generated sample
    average_annual_P = np.zeros(nYears)
    discounted_benefit = np.zeros(nSamples)
    yrs_inertia_met = np.zeros(nSamples)
    yrs_Pcrit_met = np.zeros(nSamples)
    lake_state = np.zeros(nYears + 1)

    # Randomly generate nSamples of nYears of natural P inflows
    natFlow = np.zeros((nSamples, nYears))
    for i in range(nSamples):
        np.random.seed(seed + i)
        natFlow[i, :] = np.exp(np.random.normal(log_mu, log_sigma, nYears))

    # Determine centers, radii and weights of RBFs
    C = vars[0::3]
    R = vars[1::3]
    W = vars[2::3]
    newW = np.zeros(len(W))

    # Normalize weights to sum to 1
    total = 0
    for i in range(len(W)):
        total += W[i]
    if total != 0.0:
        for i in range(len(W)):
            newW[i] = W[i] / total
    else:
        for i in range(len(W)):
            newW[i] = 1 / n

    # Run lake model simulation
    for s in range(nSamples):
        lake_state[0] = lake0
        Y = np.zeros(nYears)

        # find policy-derived emission
        temp = 0
        for i in range(len(C)):
            if R[i] != 0:
                temp = temp + W[i] * (
                    (np.absolute(lake_state[0] - C[i]) / R[i])**3)

        Y[0] = min(0.1, max(temp, 0.01))

        for i in range(nYears):
            lake_state[i + 1] = lake_state[i] * (1 - b) + (
                lake_state[i]**q) / (1 +
                                     (lake_state[i]**q)) + Y[i] + natFlow[s, i]
            average_annual_P[
                i] = average_annual_P[i] + lake_state[i + 1] / nSamples
            discounted_benefit[
                s] = discounted_benefit[s] + alpha * Y[i] * delta**i

            if i >= 1 and ((Y[i] - Y[i - 1]) > inertia_threshold):
                yrs_inertia_met[s] = yrs_inertia_met[s] + 1

            if lake_state[i + 1] < Pcrit:
                yrs_Pcrit_met[s] = yrs_Pcrit_met[s] + 1

            if i < (nYears - 1):
                # find policy-derived emission
                temp = 0
                for j in range(len(C)):
                    if R[j] != 0:
                        temp = temp + W[j] * (
                            (np.absolute(lake_state[i + 1] - C[j]) / R[j])**3)
                        Y[i + 1] = min(0.1, max(temp, 0.01))

    # Calculate minimization objectives (defined in comments at beginning of file)
    objs[0] = -1 * np.mean(discounted_benefit)  # average economic benefit
    # maximum average annual P concentration
    objs[1] = np.max(average_annual_P)
    objs[2] = -1 * np.sum(yrs_inertia_met) / (
        (nYears - 1) * nSamples
    )  # average pct of transitions meeting inertia thershold
    objs[3] = -1 * np.sum(yrs_Pcrit_met) / (nYears *
                                            nSamples)  # average reliability

    constrs = [0.0] * 1
    constrs[0] = max(0.0, reliability_threshold - (-1 * objs[3]))

    return (objs, constrs)


def lake_fast(*vars):
    def fun(x):
        return [(x[0]**2.0) / (1 + x[0]**2.0) - 0.42 * x[0]]

    soln = root(fun, 0.75)
    pcrit = soln.x

    variables = List()
    [variables.append(x) for x in vars[:]]

    return calc_internal(1234, pcrit, variables)
