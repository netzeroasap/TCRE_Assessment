import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr


def resolve_sigma(name, spec, default_scale=10.0, dims=None):
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        scale = spec.get("sigma", default_scale)
        return pm.HalfNormal(name, sigma=scale, dims=dims)
    return pm.HalfNormal(name, sigma=default_scale, dims=dims)



def build_vectorized_process_model(
    X_m,
    lookup_table=None,
    additive_processes=None,
    multiplicative_processes=None,
    noprocess_prior=None,
    sigma_struct=20.0,
    rho=None,
    sigma_process_struct=None,
    sigma_mult_struct=None,
    delta_priors=None,
    eta_priors=None,
    likelihood_noise=5.0,
    var_name="X"
):
    """
    Vectorized PyMC model for additive and multiplicative processes.
    """

    if additive_processes is None:
        additive_processes=[]
    if multiplicative_processes is None:
        multiplicative_processes = []
    if sigma_process_struct is None:
        sigma_process_struct = {}
    if sigma_mult_struct is None:
        sigma_mult_struct = {}
    if delta_priors is None:
        delta_priors = {}
    if eta_priors is None:
        eta_priors = {}

    M = X_m.shape[0]
    N_add = len(additive_processes)
    N_mult = len(multiplicative_processes)
    if lookup_table is not None:
        L_add = np.asarray(lookup_table.sel(process=additive_processes)).astype(float)
        L_mult = np.asarray(lookup_table.sel(process=multiplicative_processes)).astype(float)
    # # convert lookup table to float
    # L = np.asarray(lookup_table).astype(float)
    # # split lookup matrices
    # # Convert lookup_table (M x P) to dict {process_name: column}
    # process_names = additive_processes + multiplicative_processes
    # L_dict = {name: L[:, i] for i, name in enumerate(process_names)}
    
    # # Then build L_add and L_mult
    # L_add = pt.stack([L_dict[p] for p in additive_processes]) if N_add>0 else pt.zeros((0,M))
    # L_mult = pt.stack([L_dict[p] for p in multiplicative_processes]) if N_mult>0 else pt.zeros((0,M))

    coords = {
        "model": X_m.model.values,
        "additive_process": additive_processes,
        "multiplicative_process": multiplicative_processes,
    }

    with pm.Model(coords=coords) as model:

        # -------------------------------
        # Baseline
        # -------------------------------
        if noprocess_prior is None:
            X_unscaled = pm.Normal(f"{var_name}_unscaled", 0, 100)
        else:
            X_unscaled = noprocess_prior[var_name](f"{var_name}_unscaled")

        # -------------------------------
        # Structural variance and bias
        # -------------------------------
        sigma_struct_var = resolve_sigma("sigma_struct", sigma_struct, 20.0)

        if isinstance(rho, (int, float)):
            rho_var = float(rho)
        else:
            prior = rho if isinstance(rho, dict) else {}
            rho_var = pm.Beta("rho", prior.get("alpha", 5), prior.get("beta", 5))

        sigma_common = pm.Deterministic("sigma_common", sigma_struct_var * pt.sqrt(rho_var))
        sigma_m = pm.Deterministic("sigma_m", sigma_struct_var * pt.sqrt(1 - rho_var))

        bias_m_raw = pm.Normal("bias_m_raw", 0, 1, dims="model")
        bias_m = pm.Deterministic("bias_m", bias_m_raw * sigma_m)

        bias_common_raw = pm.Normal("bias_common_raw", 0, 1)
        bias_common = pm.Deterministic("bias_common", bias_common_raw * sigma_common)

        # ====================================================
        # ADDITIVE PROCESSES
        # ====================================================
        if N_add > 0:
            delta_list = []
            for p in additive_processes:
                prior = delta_priors.get(p, {"mu": 0, "sigma": 50})  # default
                if callable(prior):
                    # Use the callable directly
                    delta_p = prior(p)
                else:
                    # Use dict-style specification
                    mu = prior.get("mu", 0)
                    sigma = prior.get("sigma", 50)
                    delta_p = pm.Normal(p, mu=mu, sigma=sigma)
                delta_list.append(delta_p)
            # stack into a single tensor variable with dims
            delta = pm.Deterministic("delta", pt.stack(delta_list), dims="additive_process")
            # delta_mu = np.array([delta_priors.get(p, {}).get("mu", 0) for p in additive_processes])
            # delta_sigma = np.array([delta_priors.get(p, {}).get("sigma", 50) for p in additive_processes])

            # delta = pm.Normal("delta", mu=delta_mu, sigma=delta_sigma, dims="additive_process")  # (N_add,)

            sigma_add = np.array([sigma_process_struct.get(p, 5.0) for p in additive_processes])
            eps_add = pm.Normal("eps_add", 0, sigma_add[:, None], dims=("additive_process", "model"))  # (N_add, M)

            delta_pm = delta[:, None] + eps_add  # (N_add, M)
            delta_eff = pt.sum(L_add * delta_pm, axis=0)  # (M,)
        else:
            delta_eff = 0.0

        # ====================================================
        # MULTIPLICATIVE PROCESSES
        # ====================================================
        if N_mult > 0:
            eta_mu = np.array([eta_priors.get(p, {}).get("mu", 0.0) for p in multiplicative_processes])
            eta_sigma = np.array([eta_priors.get(p, {}).get("sigma", 0.2) for p in multiplicative_processes])

            # log_eta = pm.Normal("log_eta", mu=eta_mu, sigma=eta_sigma, dims="multiplicative_process")  # (N_mult,)
            # eta = pm.Deterministic("eta", pt.exp(log_eta),dims="multiplicative_process")
            log_eta_offset = pm.Normal("log_eta_offset", 0, 1, dims="multiplicative_process")
            log_eta = pm.Deterministic("log_eta", eta_mu + log_eta_offset * eta_sigma)
            eta = pm.Deterministic("eta", pt.exp(log_eta))
            
            sigma_mult = np.array([sigma_mult_struct.get(p, 0.05) for p in multiplicative_processes])
            eps_mult = pm.Normal("eps_mult", 0, sigma_mult[:, None], dims=("multiplicative_process", "model"))  # (N_mult, M)

            log_eta_pm = log_eta[:, None] + eps_mult  # (N_mult, M)
            log_mult_eff = pt.sum(L_mult * log_eta_pm, axis=0)  # (M,)
            mult_eff = pm.Deterministic("mult_eff", pt.exp(log_mult_eff), dims="model")
        else:
            mult_eff = 1.0

        # ====================================================
        # Model expectation and likelihood
        # ====================================================
        mu = X_unscaled * mult_eff + delta_eff + bias_common + bias_m

        pm.Normal("X_obs", mu=mu, sigma=likelihood_noise, observed=X_m, dims="model")
        # ----------------------------------------------------
        # Posterior deterministic for scalar X_true
        # ----------------------------------------------------
        if N_mult > 0:
            mult_factor = pt.prod(eta)  # product over all multiplicative processes
        else:
            mult_factor = 1.0
        
        if N_add > 0:
            add_factor = pt.sum(delta)  # sum over all additive processes
        else:
            add_factor = 0.0
        
        X_true = pm.Deterministic(
            var_name,
            X_unscaled * mult_factor + add_factor
        )
    return model

def add_process_information(
    model,
    process_name,
    process_data,
    process_sigma=0.1,
    process_iteration=None
):
    """
    Add observed information about a process (multiplicative or additive)
    as a pseudo-observation in the model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model.
    process_name : str
        Name of the process to update.
    process_data : float or array
        Observed value(s) for the process.
    process_sigma : float
        Observation noise for the process data.
    process_iteration : int, optional
        Optional label suffix for iterative updates.
    """

    if process_name in model.coords.get('multiplicative_process', []):
        proc_idx = model.coords['multiplicative_process'].index(process_name)
        label = f"eta_{process_name}" if process_iteration is None else f"eta_{process_name}_{process_iteration}"

        pm.Normal(
            label,
            mu=model["eta"][proc_idx],
            sigma=process_sigma,
            observed=process_data
        )

    elif process_name in model.coords.get('additive_process', []):
        proc_idx = model.coords['additive_process'].index(process_name)
        label = f"delta_{process_name}" if process_iteration is None else f"delta_{process_name}_{process_iteration}"

        pm.Normal(
            label,
            mu=model["delta"][proc_idx],
            sigma=process_sigma,
            observed=process_data
        )

    else:
        raise ValueError(f"Process '{process_name}' not found in model coordinates.")