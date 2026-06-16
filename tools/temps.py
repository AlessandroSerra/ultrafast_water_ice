from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple

import numpy as np
from numba import njit

# ==================================================================
# ------------------------- COSTANTI -------------------------------
# ==================================================================

DEFAULT_KB_CONSTANT: float = 8.31446261815324e-7
TRIG_EPSILON: float = 1e-12
DEFAULT_EPSILON: float = 1e-40
DEFAULT_HB_OO_DIST_CUTOFF: float = 3.5

# ==================================================================
# ---------------- FUNZIONI JIT OTTIMIZZATE ------------------------
# ==================================================================


@njit(fastmath=False, cache=True)
def _normalize_vector_numba(v_np: np.ndarray, epsilon_np: float) -> np.ndarray:
    """Helper function to normalize a vector, used for librations."""
    norm_val = np.linalg.norm(v_np)
    if norm_val < epsilon_np:
        return np.zeros_like(v_np)
    return v_np / norm_val


@njit(fastmath=False, cache=True)
def _process_single_frame_numba_jit(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    molecule_indices: np.ndarray,
    is_H_excited_mask: np.ndarray,
    oxygen_indices_for_hb: np.ndarray,
    kb_const: float,
    hb_cutoff_const: float,
    epsilon_const: float,
    trig_epsilon_const: float,
) -> Tuple:
    """
    Worker Numba JITtato per processare i dati di un singolo frame.
    Questa versione contiene la logica di calcolo COMPLETA E CORRETTA.
    """
    # Inizializza somme e contatori
    sum_T_stretch_exc, count_T_stretch_exc = 0.0, 0
    sum_T_stretch_norm, count_T_stretch_norm = 0.0, 0
    sum_T_bend_exc, count_T_bend_exc = 0.0, 0
    sum_T_bend_norm, count_T_bend_norm = 0.0, 0
    sum_T_bend_eq5_norm, count_T_bend_eq5_norm = 0.0, 0
    sum_T_bend_eq5_exc, count_T_bend_eq5_exc = 0.0, 0
    sum_T_bend_I_norm, count_T_bend_I_norm = 0.0, 0
    sum_T_bend_I_exc, count_T_bend_I_exc = 0.0, 0
    sum_T_hb, count_T_hb = 0.0, 0
    sum_T_twist_norm, count_T_twist_norm = 0.0, 0
    sum_T_twist_exc, count_T_twist_exc = 0.0, 0
    sum_T_wag_norm, count_T_wag_norm = 0.0, 0
    sum_T_wag_exc, count_T_wag_exc = 0.0, 0
    sum_T_rock_norm, count_T_rock_norm = 0.0, 0
    sum_T_rock_exc, count_T_rock_exc = 0.0, 0
    sum_T_cm_norm, count_T_cm_norm = 0.0, 0
    sum_T_cm_exc, count_T_cm_exc = 0.0, 0

    for mol_idx in range(molecule_indices.shape[0]):
        mol_atom_idxs = molecule_indices[mol_idx]
        O_gidx, H1_gidx, H2_gidx = mol_atom_idxs[0], mol_atom_idxs[1], mol_atom_idxs[2]

        m_O, m_H1, m_H2 = masses[O_gidx], masses[H1_gidx], masses[H2_gidx]
        M_total = m_O + m_H1 + m_H2

        # Stretch
        mu_OH1 = (m_O * m_H1) / (m_O + m_H1)
        d_OH1_vec = positions[O_gidx] - positions[H1_gidx]
        d_OH1_mag = np.linalg.norm(d_OH1_vec)
        if d_OH1_mag >= epsilon_const:
            u_OH1 = d_OH1_vec / d_OH1_mag
            v_rel1 = velocities[O_gidx] - velocities[H1_gidx]
            v_s1 = np.dot(v_rel1, u_OH1)
            T1 = (mu_OH1 * v_s1**2) / kb_const
            if is_H_excited_mask[H1_gidx]:
                sum_T_stretch_exc += T1
                count_T_stretch_exc += 1
            else:
                sum_T_stretch_norm += T1
                count_T_stretch_norm += 1

        mu_OH2 = (m_O * m_H2) / (m_O + m_H2)
        d_OH2_vec = positions[O_gidx] - positions[H2_gidx]
        d_OH2_mag = np.linalg.norm(d_OH2_vec)
        if d_OH2_mag >= epsilon_const:
            u_OH2 = d_OH2_vec / d_OH2_mag
            v_rel2 = velocities[O_gidx] - velocities[H2_gidx]
            v_s2 = np.dot(v_rel2, u_OH2)
            T2 = (mu_OH2 * v_s2**2) / kb_const
            if is_H_excited_mask[H2_gidx]:
                sum_T_stretch_exc += T2
                count_T_stretch_exc += 1
            else:
                sum_T_stretch_norm += T2
                count_T_stretch_norm += 1

        # Bending and Librations
        d_O_H1_v = positions[H1_gidx] - positions[O_gidx]
        d_O_H2_v = positions[H2_gidx] - positions[O_gidx]
        d_O_H1_m = np.linalg.norm(d_O_H1_v)
        d_O_H2_m = np.linalg.norm(d_O_H2_v)

        if d_O_H1_m >= epsilon_const and d_O_H2_m >= epsilon_const:
            u_O_H1 = d_O_H1_v / d_O_H1_m
            u_O_H2 = d_O_H2_v / d_O_H2_m

            v_O = velocities[O_gidx]
            v_H1 = velocities[H1_gidx]
            v_H2 = velocities[H2_gidx]

            # Bend (vels)
            v_H1_perp = v_H1 - np.dot(v_H1, u_O_H1) * u_O_H1
            v_H2_perp = v_H2 - np.dot(v_H2, u_O_H2) * u_O_H2
            d_H1H2_v = positions[H1_gidx] - positions[H2_gidx]
            d_H1H2_m = np.linalg.norm(d_H1H2_v)
            if d_H1H2_m >= epsilon_const:
                u_H1H2 = d_H1H2_v / d_H1H2_m
                delta_v_perp = v_H1_perp - v_H2_perp
                v_bend_scalar = np.dot(delta_v_perp, u_H1H2)
                mu_HH = (m_H1 * m_H2) / (m_H1 + m_H2)
                T_bend = (mu_HH * v_bend_scalar**2) / kb_const
                if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                    sum_T_bend_exc += T_bend
                    count_T_bend_exc += 1
                else:
                    sum_T_bend_norm += T_bend
                    count_T_bend_norm += 1

            # Bend (angle)
            cos_theta = np.dot(u_O_H1, u_O_H2)
            cos_theta = min(1.0, max(-1.0, cos_theta))
            theta = np.arccos(cos_theta)
            sin_theta = np.sin(theta)

            if abs(sin_theta) > trig_epsilon_const and M_total > epsilon_const:
                d_dot_OH1 = np.dot(v_H1 - v_O, u_O_H1)
                d_dot_OH2 = np.dot(v_H2 - v_O, u_O_H2)
                u_dot_OH1 = ((v_H1 - v_O) - u_O_H1 * d_dot_OH1) / d_O_H1_m
                u_dot_OH2 = ((v_H2 - v_O) - u_O_H2 * d_dot_OH2) / d_O_H2_m
                theta_dot = (
                    -(np.dot(u_dot_OH1, u_O_H2) + np.dot(u_O_H1, u_dot_OH2)) / sin_theta
                )

                # Bend from Eq5
                theta_half = theta / 2.0
                c_h, s_h = np.cos(theta_half), np.sin(theta_half)
                term_mH_over_M = m_H1 / M_total
                term_mO_over_M = m_O / M_total
                v_O_y_dot = term_mH_over_M * (
                    (d_dot_OH1 * c_h - d_O_H1_m * s_h * (theta_dot / 2.0))
                    + (d_dot_OH2 * c_h - d_O_H2_m * s_h * (theta_dot / 2.0))
                )
                v_H1_x_dot = d_dot_OH1 * s_h + d_O_H1_m * c_h * (theta_dot / 2.0)
                v_H1_y_dot = -term_mO_over_M * (
                    d_dot_OH1 * c_h - d_O_H1_m * s_h * (theta_dot / 2.0)
                )
                v_H2_x_dot = d_dot_OH2 * s_h + d_O_H2_m * c_h * (theta_dot / 2.0)
                v_H2_y_dot = -term_mO_over_M * (
                    d_dot_OH2 * c_h - d_O_H2_m * s_h * (theta_dot / 2.0)
                )
                K_bend_eq5 = 0.5 * (
                    m_O * (v_O_y_dot**2)
                    + m_H1 * (v_H1_x_dot**2 + v_H1_y_dot**2)
                    + m_H2 * (v_H2_x_dot**2 + v_H2_y_dot**2)
                )
                T_bend_eq5 = K_bend_eq5 / (3.0 / 2.0 * kb_const)
                if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                    sum_T_bend_eq5_exc += T_bend_eq5
                    count_T_bend_eq5_exc += 1
                else:
                    sum_T_bend_eq5_norm += T_bend_eq5
                    count_T_bend_eq5_norm += 1

                # Bend from Inertia
                d_OH_equil = 0.9572
                I_mom_HH = m_H1 * d_OH_equil**2 + m_H2 * d_OH_equil**2
                K_bend_I = 0.5 * I_mom_HH * (theta_dot**2)
                T_bend_I = 0.5 * K_bend_I / kb_const
                if is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]:
                    sum_T_bend_I_exc += T_bend_I
                    count_T_bend_I_exc += 1
                else:
                    sum_T_bend_I_norm += T_bend_I
                    count_T_bend_I_norm += 1

                # Librations
                # ========= INIZIO BLOCCO CORRETTO =========
                masses_mol = np.empty(3, dtype=np.float64)
                masses_mol[0] = m_O
                masses_mol[1] = m_H1
                masses_mol[2] = m_H2

                positions_mol = np.empty((3, 3), dtype=np.float64)
                positions_mol[0, :] = positions[O_gidx]
                positions_mol[1, :] = positions[H1_gidx]
                positions_mol[2, :] = positions[H2_gidx]

                velocities_mol = np.empty((3, 3), dtype=np.float64)
                velocities_mol[0, :] = velocities[O_gidx]
                velocities_mol[1, :] = velocities[H1_gidx]
                velocities_mol[2, :] = velocities[H2_gidx]
                # ========= FINE BLOCCO CORRETTO =========

                R_cm = (
                    positions_mol[0] * m_O
                    + positions_mol[1] * m_H1
                    + positions_mol[2] * m_H2
                ) / M_total
                V_cm = (
                    velocities_mol[0] * m_O
                    + velocities_mol[1] * m_H1
                    + velocities_mol[2] * m_H2
                ) / M_total

                r_prime_lib_val = positions_mol - R_cm
                v_rel_cm_lib_val = velocities_mol - V_cm

                axis1_u = _normalize_vector_numba(u_O_H1 + u_O_H2, epsilon_const)
                axis3_w = _normalize_vector_numba(
                    np.cross(u_O_H1, u_O_H2), epsilon_const
                )
                axis2_v = _normalize_vector_numba(
                    np.cross(axis3_w, axis1_u), epsilon_const
                )

                axis3_w = _normalize_vector_numba(
                    np.cross(axis1_u, axis2_v), epsilon_const
                )

                I_11, I_22, I_33 = 0.0, 0.0, 0.0
                for i_atom in range(3):
                    r_p_i = r_prime_lib_val[i_atom]
                    m_i = masses_mol[i_atom]
                    r_p_dot_u, r_p_dot_v, r_p_dot_w = (
                        np.dot(r_p_i, axis1_u),
                        np.dot(r_p_i, axis2_v),
                        np.dot(r_p_i, axis3_w),
                    )
                    I_11 += m_i * (r_p_dot_v**2 + r_p_dot_w**2)
                    I_22 += m_i * (r_p_dot_u**2 + r_p_dot_w**2)
                    I_33 += m_i * (r_p_dot_u**2 + r_p_dot_v**2)

                L_lab = np.zeros(3, dtype=np.float64)
                for i_atom in range(3):
                    L_lab += masses_mol[i_atom] * np.cross(
                        r_prime_lib_val[i_atom], v_rel_cm_lib_val[i_atom]
                    )

                L1, L2, L3 = (
                    np.dot(L_lab, axis1_u),
                    np.dot(L_lab, axis2_v),
                    np.dot(L_lab, axis3_w),
                )
                mol_is_excited = (
                    is_H_excited_mask[H1_gidx] or is_H_excited_mask[H2_gidx]
                )

                if I_11 > epsilon_const:
                    T_twist = L1**2 / (I_11 * kb_const)
                    if mol_is_excited:
                        sum_T_twist_exc += T_twist
                        count_T_twist_exc += 1
                    else:
                        sum_T_twist_norm += T_twist
                        count_T_twist_norm += 1
                if I_22 > epsilon_const:
                    T_rock = L2**2 / (I_22 * kb_const)
                    if mol_is_excited:
                        sum_T_rock_exc += T_rock
                        count_T_rock_exc += 1
                    else:
                        sum_T_rock_norm += T_rock
                        count_T_rock_norm += 1
                if I_33 > epsilon_const:
                    T_wag = L3**2 / (I_33 * kb_const)
                    if mol_is_excited:
                        sum_T_wag_exc += T_wag
                        count_T_wag_exc += 1
                    else:
                        sum_T_wag_norm += T_wag
                        count_T_wag_norm += 1

                # Center of Mass Temperature
                V_cm_mag_sq = np.dot(V_cm, V_cm)
                K_cm = 0.5 * M_total * V_cm_mag_sq
                T_cm = K_cm / (1.5 * kb_const)
                if mol_is_excited:
                    sum_T_cm_exc += T_cm
                    count_T_cm_exc += 1
                else:
                    sum_T_cm_norm += T_cm
                    count_T_cm_norm += 1

    # Hydrogen Bond
    num_oxygens = len(oxygen_indices_for_hb)
    for i_o in range(num_oxygens):
        for j_o in range(i_o + 1, num_oxygens):
            O1_idx, O2_idx = oxygen_indices_for_hb[i_o], oxygen_indices_for_hb[j_o]
            d_OO_vec = positions[O1_idx] - positions[O2_idx]
            d_OO_mag = np.linalg.norm(d_OO_vec)
            if epsilon_const < d_OO_mag < hb_cutoff_const:
                u_OO = d_OO_vec / d_OO_mag
                v_rel_OO = velocities[O1_idx] - velocities[O2_idx]
                v_hb_scalar = np.dot(v_rel_OO, u_OO)
                mu_OO = masses[O1_idx] / 2.0
                sum_T_hb += (mu_OO * v_hb_scalar**2) / kb_const
                count_T_hb += 1

    return (
        sum_T_stretch_exc / count_T_stretch_exc if count_T_stretch_exc > 0 else np.nan,
        sum_T_stretch_norm / count_T_stretch_norm
        if count_T_stretch_norm > 0
        else np.nan,
        sum_T_bend_exc / count_T_bend_exc if count_T_bend_exc > 0 else np.nan,
        sum_T_bend_norm / count_T_bend_norm if count_T_bend_norm > 0 else np.nan,
        sum_T_bend_eq5_exc / count_T_bend_eq5_exc
        if count_T_bend_eq5_exc > 0
        else np.nan,
        sum_T_bend_eq5_norm / count_T_bend_eq5_norm
        if count_T_bend_eq5_norm > 0
        else np.nan,
        sum_T_bend_I_exc / count_T_bend_I_exc if count_T_bend_I_exc > 0 else np.nan,
        sum_T_bend_I_norm / count_T_bend_I_norm if count_T_bend_I_norm > 0 else np.nan,
        sum_T_hb / count_T_hb if count_T_hb > 0 else np.nan,
        sum_T_twist_exc / count_T_twist_exc if count_T_twist_exc > 0 else np.nan,
        sum_T_twist_norm / count_T_twist_norm if count_T_twist_norm > 0 else np.nan,
        sum_T_wag_exc / count_T_wag_exc if count_T_wag_exc > 0 else np.nan,
        sum_T_wag_norm / count_T_wag_norm if count_T_wag_norm > 0 else np.nan,
        sum_T_rock_exc / count_T_rock_exc if count_T_rock_exc > 0 else np.nan,
        sum_T_rock_norm / count_T_rock_norm if count_T_rock_norm > 0 else np.nan,
        sum_T_cm_norm / count_T_cm_norm if count_T_cm_norm > 0 else np.nan,
        sum_T_cm_exc / count_T_cm_exc if count_T_cm_exc > 0 else np.nan,
    )


# ==================================================================
# ------------ FUNZIONE WORKER PER PARALLELISMO --------------------
# ==================================================================


def _parallel_worker_from_arrays(args_tuple: Tuple) -> Tuple[int, np.ndarray]:
    """
    Worker function for parallel processing that accepts NumPy arrays directly.
    """
    try:
        (
            frame_idx,
            pos_frame,
            vel_frame,
            masses,
            molecule_indices,
            is_H_excited_mask,
            oxygen_indices_for_hb,
            kb_const,
            hb_cutoff_const,
            epsilon_const,
            trig_epsilon_const,
        ) = args_tuple

        temps_tuple = _process_single_frame_numba_jit(
            pos_frame,
            vel_frame,
            masses,
            molecule_indices,
            is_H_excited_mask,
            oxygen_indices_for_hb,
            kb_const,
            hb_cutoff_const,
            epsilon_const,
            trig_epsilon_const,
        )
        return frame_idx, np.array(temps_tuple, dtype=np.float64)
    except Exception as e:
        print(f"Errore nel worker per il frame {args_tuple[0]}: {e}")
        return args_tuple[0], np.full(17, np.nan, dtype=np.float64)


# ==================================================================
# -------------------- FUNZIONE WRAPPER UTENTE ---------------------
# ==================================================================


def analyzeTEMPS(
    positions: np.ndarray,
    velocities: np.ndarray,
    atom_types: np.ndarray,
    excited_indices_filepath: str,
    kb_constant: float = DEFAULT_KB_CONSTANT,
    hb_cutoff: float = DEFAULT_HB_OO_DIST_CUTOFF,
    epsilon_val: float = DEFAULT_EPSILON,
    trig_epsilon: float = TRIG_EPSILON,
    num_workers: int = None,
) -> Dict[str, np.ndarray]:
    """
    Analizza le temperature molecolari partendo da array NumPy, usando un
    ProcessPoolExecutor per parallelizzare il calcolo sui frame.

    ASSUNZIONE: Gli atomi sono ordinati per molecola (O,H,H, O,H,H, ...)
    e i tipi sono 1 per Ossigeno e 2 per Idrogeno.
    """
    n_frames, n_atoms, _ = positions.shape

    # --- Preparazione Dati (eseguita una sola volta nel processo principale) ---
    if n_atoms % 3 != 0:
        raise ValueError("Il numero di atomi non è divisibile per 3.")
    n_mols = n_atoms // 3
    molecule_indices = np.arange(n_atoms, dtype=np.int32).reshape(n_mols, 3)

    mass_map = np.array([0.0, 15.999, 1.008], dtype=np.float64)
    try:
        masses = mass_map[atom_types]
    except IndexError:
        raise ValueError(
            "Trovato un tipo di atomo non valido. Ammessi solo 1 (O) e 2 (H)."
        )

    try:
        loaded_indices = np.loadtxt(excited_indices_filepath, dtype=np.int64)
        excited_h_indices = loaded_indices - 1
        if excited_h_indices.ndim == 0:
            excited_h_indices = np.array([excited_h_indices.item()])
    except Exception as e:
        print(f"Attenzione: impossibile caricare gli indici eccitati. Errore: {e}")
        excited_h_indices = np.array([], dtype=np.int64)

    is_H_excited_mask = np.zeros(n_atoms, dtype=np.bool_)
    if excited_h_indices.size > 0:
        is_H_excited_mask[excited_h_indices] = True

    oxygen_indices_for_hb = np.where(atom_types == 1)[0].astype(np.int32)

    # --- Creazione dei Task per i Worker ---
    tasks_args = []
    for i in range(n_frames):
        tasks_args.append(
            (
                i,
                positions[i],
                velocities[i],
                masses,
                molecule_indices,
                is_H_excited_mask,
                oxygen_indices_for_hb,
                kb_constant,
                hb_cutoff,
                epsilon_val,
                trig_epsilon,
            )
        )

    # --- Esecuzione Parallela ---
    results_array = np.full((n_frames, 17), np.nan, dtype=np.float64)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        try:
            from tqdm import tqdm

            future_results = list(
                tqdm(
                    executor.map(_parallel_worker_from_arrays, tasks_args),
                    total=n_frames,
                    desc="Processing Frames",
                )
            )
        except ImportError:
            future_results = list(
                executor.map(_parallel_worker_from_arrays, tasks_args)
            )

    for frame_idx, result_temps_array in future_results:
        results_array[frame_idx, :] = result_temps_array

    # --- Unpack dei risultati ---
    returned_data = {
        "stretch_excited_H": results_array[:, 0],
        "stretch_normal_H": results_array[:, 1],
        "bend_HOH_exc": results_array[:, 2],
        "bend_HOH_norm": results_array[:, 3],
        "bend_HOH_eq5_exc": results_array[:, 4],
        "bend_HOH_eq5_norm": results_array[:, 5],
        "bend_HOH_I_exc": results_array[:, 6],
        "bend_HOH_I_norm": results_array[:, 7],
        "hb": results_array[:, 8],
        "libr_twist_exc": results_array[:, 9],
        "libr_twist_norm": results_array[:, 10],
        "libr_wag_exc": results_array[:, 11],
        "libr_wag_norm": results_array[:, 12],
        "libr_rock_exc": results_array[:, 13],
        "libr_rock_norm": results_array[:, 14],
        "cm_translation_norm": results_array[:, 15],
        "cm_translation_exc": results_array[:, 16],
    }

    return returned_data
