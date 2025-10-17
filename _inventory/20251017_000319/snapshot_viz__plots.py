"""
Figures for Phase 1 experiments (updated model).

Implements:
- capital_dilution_plot: (Y/L)_D and (Y/L)_L over time for a demographic/migration shock
- emissions_and_ratio_plot: world E_t and ratio M_{t+1}/E_t (plotted against t)
- ambiguity_map: interior-slope sign boundary at xi * y = 1/2 (eta annotated only)

All functions save .png files into the 'figures/' folder with standardized names.
They expect SimResult objects (from simulate) and/or structured experiment outputs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Use model helper for the recorded ratio limit (keeps Companion truthful, tests/audit consistent)
try:
    from model.solow import carbon_ratio_recorded_limit  # (1+g_E)/((1+g_E) - phi_M)
except Exception:
    carbon_ratio_recorded_limit = None  # graceful fallback if not available


def _ensure_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    If 'path' is already a file path (has extension), create its parent.
    If 'path' is a directory name, create that.
    """
    root, ext = os.path.splitext(path)
    dir_to_make = os.path.dirname(path) if ext else (path or ".")
    os.makedirs(dir_to_make or ".", exist_ok=True)


# ------------------------------------------------
# 1) Capital dilution plots (Y/L paths)
# ------------------------------------------------

def capital_dilution_plot(res_base, res_shock, shock_name: str, outdir: str = "figures") -> None:
    """
    Plot (Y/L)_D and (Y/L)_L against time for a demographic shock (nD_up, nL_up)
    or migration easing (which shifts active-labor growth endogenously).

    Uses the precomputed productivity series q_t = (Y/L)_t from CountrySeries.q.
    """
    _ensure_dir(outdir)

    # Time grid for flows: t = 0..T-1
    t = res_base.tgrid[:-1]

    # Productivity per active worker q = Y/L (already computed in simulate)
    qD_base = np.asarray(res_base.D.q[:-1], dtype=float)
    qL_base = np.asarray(res_base.L.q[:-1], dtype=float)
    qD_shock = np.asarray(res_shock.D.q[:-1], dtype=float)
    qL_shock = np.asarray(res_shock.L.q[:-1], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Developed (D)
    ax[0].plot(t, qD_base, label="Baseline D")
    ax[0].plot(t, qD_shock, linestyle="--", label="Shock D")
    ax[0].set_title("(Y/L)_D over time")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Output per active worker (Y/L)")
    ax[0].legend()

    # Developing (L)
    ax[1].plot(t, qL_base, label="Baseline L")
    ax[1].plot(t, qL_shock, linestyle="--", label="Shock L")
    ax[1].set_title("(Y/L)_L over time")
    ax[1].set_xlabel("t")
    ax[1].legend()

    fig.suptitle(f"Capital dilution under {shock_name}")
    # Reserve space for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(outdir, f"fig_capital_dilution_{shock_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ------------------------------------------------
# 2) Emissions and ratio plots
# ------------------------------------------------

def emissions_and_ratio_plot(
    res_base,
    res_shock,
    shock_name: str,
    phi_M: float = None,
    outdir: str = "figures",
    show_asymptote: bool = False,
) -> None:
    """
    Plot world emissions E_t (flows) and the recorded ratio M_{t+1}/E_t.

    Optional horizontal asymptotes (right panel) are computed from tail g_E:
        M_{t+1}/E_t -> (1+g_E)/((1+g_E) - phi_M)
    using the model helper carbon_ratio_recorded_limit(...) when available.

    If phi_M is not provided, estimate it from the baseline tail via:
        phi_M ≈ median_tail( (M_{t+1} - E_t) / M_t )
    """
    _ensure_dir(outdir)

    t_flows = res_base.tgrid[:-1]   # for E_t (length T)
    t_ratio = res_base.tgrid        # for M_{t+1}/E_t (length T+1; typically starts with NaN)

    E_base  = np.asarray(res_base.E_world, float)
    E_shock = np.asarray(res_shock.E_world, float)
    MoverE_base  = np.asarray(res_base.M_over_E, float)   # recorded ratio
    MoverE_shock = np.asarray(res_shock.M_over_E, float)  # recorded ratio

    def _tail_g(x, tail=20):
        x = np.asarray(x, float)
        if x.size < 2:
            return np.nan
        k = min(tail, x.size - 1)
        seg = x[-(k+1):]
        if np.any(seg <= 0):
            return np.nan
        return float(np.mean(np.log(seg[1:] / seg[:-1])))

    # If phi_M is not passed, estimate it from the recursion using the baseline tail.
    if phi_M is None:
        Mp, Mn = np.asarray(res_base.M[:-1], float), np.asarray(res_base.M[1:], float)
        Ep     = E_base  # aligned with Mp -> Mn
        denom  = np.clip(Mp, 1e-12, None)
        phi_series = (Mn - Ep) / denom
        k = min(30, phi_series.size)
        phi_M_est = float(np.nanmedian(phi_series[-k:])) if phi_series.size else np.nan
        phi_use = phi_M_est
    else:
        phi_use = float(phi_M)

    # Prepare figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # --- Emissions panel
    ax[0].plot(t_flows, E_base, label="Baseline")
    ax[0].plot(t_flows, E_shock, linestyle="--", label="Shock")
    ax[0].set_title("World emissions $E_t$")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Emissions (physical units)")
    ax[0].legend()

    # --- Ratio panel (recorded: M_{t+1}/E_t)
    ax[1].plot(t_ratio, MoverE_base, label="Baseline $M_{t+1}/E_t$")
    ax[1].plot(t_ratio, MoverE_shock, linestyle="--", label="Shock $M_{t+1}/E_t$")
    ax[1].set_title("Recorded ratio $M_{t+1} / E_t$")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("$M/E$")

    # Optional horizontal asymptotes (if computable)
    if show_asymptote:
        gB = _tail_g(E_base)
        gS = _tail_g(E_shock)

        def _limit(g):
            if not np.isfinite(g) or not np.isfinite(phi_use):
                return np.nan
            # Prefer model helper if available (raises if condition fails)
            if carbon_ratio_recorded_limit is not None:
                try:
                    return carbon_ratio_recorded_limit(g, phi_use)
                except Exception:
                    return np.nan
            # Fallback: compute directly
            if (1.0 + g) <= phi_use:
                return np.nan  # condition violated -> no finite limit
            return (1.0 + g) / ((1.0 + g) - phi_use)

        limB = _limit(gB)
        limS = _limit(gS)

        if np.isfinite(limB):
            ax[1].axhline(limB, linestyle=":", linewidth=1.6,
                          label=f"Baseline limit ≈ {limB:.3f}")
        if np.isfinite(limS):
            ax[1].axhline(limS, linestyle=":", linewidth=1.6,
                          label=f"Shock limit ≈ {limS:.3f}")

        # Put a small note inside the axes to avoid cropping
        foot = []
        if np.isfinite(gB): foot.append(f"g_E(base)≈{gB:.4f}")
        if np.isfinite(gS): foot.append(f"g_E(shock)≈{gS:.4f}")
        if np.isfinite(phi_use): foot.append(f"φ_M≈{phi_use:.3f}")
        if foot:
            ax[1].text(0.02, 0.02, "; ".join(foot), transform=ax[1].transAxes, fontsize=9)

    ax[1].legend(loc="best")

    fig.suptitle(f"Emissions and ratio under {shock_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(outdir, f"fig_emissions_ratio_{shock_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ------------------------------------------------
# 3) Ambiguity map (xi*y = 1/2 boundary)
# ------------------------------------------------

def ambiguity_map(xi_vals, y_vals, eta: float, outdir: str = "figures") -> None:
    """
    Plot the interior-slope sign boundary for per-capita emissions when
    epsilon depends on productivity q = Y/L, and abatement is interior (theta = xi * y):

      e ∝ q^eta * y * (1 - xi*y)   ⇒
      ∂ln e / ∂ln y |_q = 1 - (xi*y)/(1 - xi*y)

    The sign flips at xi*y = 1/2 (independent of eta). We shade the region
    xi*y < 1/2 (positive slope) vs xi*y > 1/2 (negative slope), and annotate eta.
    """
    _ensure_dir(outdir)

    X, Y = np.meshgrid(np.asarray(xi_vals, dtype=float), np.asarray(y_vals, dtype=float))
    boundary = 0.5  # xi*y = 1/2

    # Positive-slope region indicator (True where xi*y < 1/2)
    pos_region = (X * Y) < boundary

    fig, ax = plt.subplots(figsize=(6, 5))

    # Two-region fill: positive-slope (light) vs negative-slope (darker)
    ax.contourf(
        X, Y, pos_region.astype(float),
        levels=[-0.5, 0.5, 1.5],
        colors=["lightgray", "white"]
    )

    # Boundary curve xi*y = 1/2
    C = ax.contour(X, Y, X * Y, levels=[boundary], linewidths=2)

    # Robust legend (works across Matplotlib versions)
    try:
        C.collections[0].set_label(r"Boundary: $\xi y=\frac{1}{2}$")
        ax.legend(loc="best")
    except Exception:
        from matplotlib.lines import Line2D
        proxy = Line2D([0], [0], linestyle='-', linewidth=2)
        ax.legend([proxy], [r"Boundary: $\xi y=\frac{1}{2}$"], loc="best")

    # Title and labels
    ax.set_title(f"Ambiguity map (η annotated: {eta:.2f})")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$y$")

    # Enforce filename with two-decimal eta tag to keep naming stable
    eta_tag = f"{eta:.2f}"
    fname = os.path.join(outdir, f"fig_ambiguity_eta-{eta_tag}.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)

