from typing import Optional, Tuple

import numpy as np
from numba import njit

# Costanti utili (possono essere definite globalmente o passate/definite localmente)
TWO_PI = 2.0 * np.pi
_ATOMIC_MASSES = np.array(
    [0.0, 15.999, 1.008], dtype=np.float32
)  # indice 1=O, 2=H, 0 dummy

# Fattore per convertire frequenza (in ps^-1 o THz) in wavenumbers (cm^-1)
# 1 / (c_luce_cm_per_s * 1e-12 s/ps) = 1 / (2.99792458e10 * 1e-12)
HZ_TO_CMINV_FACTOR = 33.3564095198152


def _vacf_fft_core(
    velocities: np.ndarray,  # (n_frames, n_atoms_grp, 3)
    masses: Optional[np.ndarray],  # (n_atoms_grp,) oppure None se non pesata
    group: str,
    corr_len: Optional[int],
    mass_weighted: bool,
) -> np.ndarray:
    vel = np.asarray(velocities, dtype=np.float64)
    n_frames, n_atoms_grp, ndim = vel.shape

    if corr_len > n_frames:
        raise ValueError("max_correlation_len > n_frames")

    vel_flat = vel.reshape(n_frames, -1)  # (n_frames, n_atoms_grp*3)

    if mass_weighted:
        if masses is None:
            raise ValueError("mass_weighted=True ma masses è None")

        m = np.asarray(masses, dtype=np.float64)
        if m.ndim != 1:
            raise ValueError("masses deve essere un array 1D di lunghezza n_atoms_grp")

        if m.shape[0] != n_atoms_grp:
            raise ValueError("Lunghezza masses incompatibile con n_atoms_grp")

        # Ripeti ogni massa per le 3 componenti (x,y,z)
        m_series = np.repeat(m, ndim)  # (n_atoms_grp*3,)
        vel_flat *= np.sqrt(m_series[None, :])

    # FFT length (potenza di 2 >= 2*n_frames)
    n_fft = 1
    target = 2 * n_frames
    while n_fft < target:
        n_fft <<= 1

    F = np.fft.rfft(vel_flat, n=n_fft, axis=0)
    S = F * np.conjugate(F)
    acf = np.fft.irfft(S, n=n_fft, axis=0)[:corr_len].real  # (corr_len, n_series)

    # Somma su atomi*componenti, poi normalizza su C(0)
    C_t = acf.sum(axis=1)  # somma su n_series = n_atoms_grp*3
    C_t = C_t.astype(np.float32, copy=False)
    C_t /= C_t[0]

    return C_t


def calculateVACFgroup(
    velocities: np.ndarray,  # (n_frames, n_atoms_tot, 3)
    atom_types: np.ndarray,  # (n_atoms_tot,) valori interi per indicizzare _ATOMIC_MASSES
    group: str,
    index_file: str,
    max_correlation_len: Optional[int] = None,
    mass_weighted: bool = False,
    remove_com: bool = True,
) -> np.ndarray:
    # --- 0) Opzionale: rimuovi velocità del centro di massa ---
    n_frames, n_atoms, _ = velocities.shape
    corr_len = (n_frames - 1) if max_correlation_len is None else max_correlation_len

    if remove_com:
        print("Removing center of mass velocity.")
        # Calcola le masse atomiche per tutti gli atomi una sola volta
        atomic_masses_all = _ATOMIC_MASSES[atom_types]  # (n_atoms,)
        total_mass = np.sum(atomic_masses_all)

        # velocities shape: (n_frames, n_atoms, 3)
        # atomic_masses_all.reshape(1, -1, 1) shape: (1, n_atoms, 1)
        com_vel = (
            np.sum(velocities * atomic_masses_all.reshape(1, -1, 1), axis=1)
            / total_mass
        )

        # Sottrai la velocità del COM da tutti gli atomi per ogni frame (IN-PLACE)
        velocities -= com_vel.reshape(-1, 1, 3)

    # --- 1) selezione atomi (exc / norm / tutti) ---
    if group in ("exc", "norm"):
        exc_idxs = np.loadtxt(index_file, dtype=int) - 1  # 1-based → 0-based
        all_idxs = np.arange(n_atoms, dtype=int)

        if group == "exc":
            sel = exc_idxs
        else:  # "norm"
            sel = np.setdiff1d(all_idxs, exc_idxs)

        velocities_grp = velocities[:, sel, :]  # (n_frames, n_atoms_grp, 3)
        atom_types_grp = atom_types[sel]  # (n_atoms_grp,)
    elif group == "all":
        # nessun gruppo speciale → prendi tutti
        velocities_grp = velocities
        atom_types_grp = atom_types
    else:
        raise ValueError(
            f"Unknown group='{group}' specified. Use 'exc', 'norm', or 'all'."
        )

    n_atoms_grp = velocities_grp.shape[1]

    # --- 2) masse dal tipo atomico (sempre le stesse nel tempo) ---
    if mass_weighted:
        # masse per il gruppo selezionato (1D, costanti nel tempo)
        masses_grp = _ATOMIC_MASSES[atom_types_grp].astype(np.float64)  # (n_atoms_grp,)
    else:
        masses_grp = None

    print(
        f"Calculating {'mass-weighted ' if mass_weighted else ''}"
        f"VACF (FFT) for group='{group}', n_atoms={n_atoms_grp}, steps={corr_len}."
    )

    # --- 3) passa al kernel FFT che lavora su un gruppo generico ---
    C_t = _vacf_fft_core(
        velocities_grp,
        masses_grp,
        group=group,
        corr_len=corr_len,
        mass_weighted=mass_weighted,
    )
    return C_t


@njit(cache=True)  # Aggiunto cache=True per riutilizzare la compilazione
def _filon_cosine_transform_subroutine(
    dt_val: float,
    delta_omega_val: float,  # Questo è il DOM del Fortran, interpretato come Delta_Omega
    nmax_intervals: int,  # NMAX nel Fortran (numero di intervalli, deve essere pari)
    c_corr_func: np.ndarray,  # Funzione di correlazione C(t), array da 0 a nmax_intervals
    chat_spectrum: np.ndarray,  # Array di output per lo spettro CHAT(omega)
) -> None:
    """
    Replica Python della subroutine FILONC per calcolare la trasformata coseno
    di Fourier usando il metodo di Filon.

    Args:
        dt_val: Intervallo di tempo tra i punti in c_corr_func.
        delta_omega_val: Intervallo di frequenza angolare per chat_spectrum.
                         (omega_nu = nu * delta_omega_val).
        nmax_intervals: Numero di intervalli sull'asse del tempo. Deve essere pari.
                        c_corr_func deve avere nmax_intervals + 1 punti.
        c_corr_func: Array NumPy 1D della funzione di correlazione C(t).
        chat_spectrum: Array NumPy 1D (output, modificato in-place) per lo spettro.
    """
    if nmax_intervals % 2 != 0:
        raise ValueError(
            "NMAX (nmax_intervals) deve essere pari per il metodo di Filon."
        )
    if len(c_corr_func) != nmax_intervals + 1:
        raise ValueError(
            f"Lunghezza di c_corr_func ({len(c_corr_func)}) non corrisponde a "
            f"nmax_intervals + 1 ({nmax_intervals + 1})."
        )
    if len(chat_spectrum) != nmax_intervals + 1:
        raise ValueError(
            "chat_spectrum deve avere la stessa dimensione di c_corr_func."
        )

    t_max = float(nmax_intervals) * dt_val

    for nu_idx in range(nmax_intervals + 1):  # Loop su NU da 0 a NMAX
        omega = float(nu_idx) * delta_omega_val  # Frequenza angolare corrente
        theta = omega * dt_val  # Argomento adimensionale omega*dt

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if (
            abs(theta) < 1e-9
        ):  # Caso speciale per theta ~ 0 (per evitare divisione per zero)
            alpha = 0.0
            beta = 2.0 / 3.0
            gamma = 4.0 / 3.0
        else:
            th_sq = theta * theta
            th_cub = th_sq * theta
            # Parametri di Filon
            alpha = (1.0 / th_cub) * (
                th_sq + theta * sin_theta * cos_theta - 2.0 * sin_theta**2
            )
            beta = (2.0 / th_cub) * (
                theta * (1.0 + cos_theta**2) - 2.0 * sin_theta * cos_theta
            )
            gamma = (4.0 / th_cub) * (sin_theta - theta * cos_theta)

        # Somma sui termini con indice pari (CE)
        ce_sum = 0.0
        for tau_idx in range(0, nmax_intervals + 1, 2):  # tau = 0, 2, ..., NMAX
            ce_sum += c_corr_func[tau_idx] * np.cos(theta * float(tau_idx))

        # Sottrai metà del primo e ultimo termine (correzione per la regola di Filon/Simpson)
        ce_sum -= 0.5 * (
            c_corr_func[0] * np.cos(theta * 0.0)  # np.cos(0)=1
            + c_corr_func[nmax_intervals] * np.cos(theta * float(nmax_intervals))
        )
        # Nota: theta * nmax_intervals = omega * dt * nmax_intervals = omega * t_max

        # Somma sui termini con indice dispari (CO)
        co_sum = 0.0
        for tau_idx in range(1, nmax_intervals, 2):  # tau = 1, 3, ..., NMAX-1
            co_sum += c_corr_func[tau_idx] * np.cos(theta * float(tau_idx))

        # Calcola il valore dello spettro CHAT(NU)
        term_alpha_component = (
            alpha * c_corr_func[nmax_intervals] * np.sin(omega * t_max)
        )
        chat_spectrum[nu_idx] = (
            2.0 * (term_alpha_component + beta * ce_sum + gamma * co_sum) * dt_val
        )


def calculateVDOS(
    time_points: np.ndarray,
    corr_values: np.ndarray,
    norm: bool = True,
    gaussian_filter_width: Optional[float] = None,
    output_in_wavenumbers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Processa una funzione di correlazione temporale per ottenere il suo spettro di frequenza,
    replicando la logica del programma Fortran 'ft', incluso il metodo di Filon.

    Args:
        time_points: Array NumPy 1D dei punti temporali. Si assume dt costante.
        corr_values: Array NumPy 1D dei valori della funzione di correlazione.
        norm: (Opzionale) Se True, normalizza i valori della funzione di correlazione ad integrale = 1
        gaussian_filter_width: (Opzionale) Parametro 'width' per il filtro Gaussiano.
                               Se None, nessun filtro viene applicato.
        output_in_wavenumbers: (Opzionale) Se True (default), l'asse delle frequenze
                               dell'output è in cm^-1. Altrimenti, è in unità di
                               frequenza angolare (rad / unità di tempo di dt).

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            - frequencies_axis: Array delle frequenze (in cm^-1 o rad/unità di tempo).
            - spectrum: Array dello spettro calcolato (CHAT).
            - filter_info: (Opzionale) Se il filtro è applicato, una tupla contenente
                             (tempo_filtrato, valori_corr_filtrati, finestra_usata).
                             Altrimenti None.
    """
    if len(time_points) != len(corr_values):
        raise ValueError("time_points e corr_values devono avere la stessa lunghezza.")
    if (
        len(time_points) < 2
    ):  # Per Filon (NMAX pari) serve NMAX>=2, quindi almeno 3 punti.
        # Se NMAX=0 (1 punto), tmax=0, dom=inf.
        # Se NMAX<2, non ha senso.
        print(
            "Attenzione: Servono almeno 3 punti dati (2 intervalli) per il metodo di Filon "
            "come implementato. Se meno, si restituiscono array vuoti."
        )
        return np.array([]), np.array([]), None

    dt = time_points[1] - time_points[0]
    # Verifica grossolana della costanza di dt (opzionale ma buona pratica)
    # if not np.allclose(np.diff(time_points), dt):
    #     print("Attenzione: dt rilevato non costante. La precisione potrebbe essere affetta. Si usa il primo dt.")

    num_points_original = len(corr_values)

    # NMAX nel Fortran è il numero di INTERVALLI, e deve essere pari.
    # Se num_points_original è il numero di punti, n_intervals_initial = num_points_original - 1.
    nmax_intervals = num_points_original - 1

    # Lavora su una copia dei valori di correlazione
    current_corr_for_ft = np.copy(corr_values)

    if nmax_intervals < 0:  # Caso di 0 o 1 punto originale
        print("Non abbastanza punti per formare intervalli.")
        return np.array([]), np.array([]), None

    if nmax_intervals % 2 != 0:
        nmax_intervals -= 1  # Rendi nmax_intervals pari, scartando l'ultimo intervallo (e l'ultimo punto)
        num_points_to_use = nmax_intervals + 1
        current_corr_for_ft = current_corr_for_ft[:num_points_to_use]
        time_points_to_use = time_points[:num_points_to_use]
        print(
            f"Numero di intervalli reso pari: {nmax_intervals}. "
            f"Dati troncati a {num_points_to_use} punti."
        )
    else:
        num_points_to_use = nmax_intervals + 1
        time_points_to_use = (
            time_points  # Nessun troncamento necessario oltre alla parità di NMAX
        )

    if (
        nmax_intervals < 2 and nmax_intervals != 0
    ):  # Filon richiede NMAX pari >= 2, se non è NMAX=0 (1 punto, 0 intervalli)
        # Il caso NMAX=0 (1 punto) viene gestito da Filon (theta=0)
        if nmax_intervals == 0 and num_points_to_use == 1:
            pass  # NMAX=0 (1 punto) è un caso limite che Filon gestisce
        else:
            print(
                f"Attenzione: NMAX={nmax_intervals} (dopo aggiustamento parità) non è sufficiente per la procedura "
                "standard di Filon (richiede NMAX >= 2 o NMAX=0)."
            )
            return np.array([]), np.array([]), None

    t_max = float(nmax_intervals) * dt
    if (
        t_max == 0 and nmax_intervals > 0
    ):  # Dovrebbe accadere solo se dt=0, che è un problema
        raise ValueError(
            "t_max è zero con nmax_intervals > 0, implica dt=0, il che non è valido."
        )

    # DOM nel Fortran è interpretato come Delta_Omega per la subroutine FILONC.
    # delta_omega_initial = 1.0 / t_max # Questa era la riga del Fortran per DOM
    # Se t_max è 0 (caso NMAX=0, un solo punto), 1.0/t_max è infinito.
    # Per NMAX=0, t_max=0. DOM non ha molto senso, omega sarà sempre 0*DOM=0.
    # La subroutine Filon gestisce theta=0.
    if t_max > 0:
        delta_omega_for_filon = 1.0 / t_max
    else:  # Caso NMAX=0, un solo punto
        delta_omega_for_filon = 0.0  # O qualsiasi valore, omega sarà 0.

    print("\nParametri per la trasformata di Filon:")
    print(f"  Numero di punti usati (NMAX+1): {num_points_to_use}")
    print(f"  Numero di intervalli (NMAX): {nmax_intervals}")
    print(f"  dt   = {dt:.4e}")
    print(f"  tmax = {t_max:.4e}")
    print(
        f"  Delta Omega (DOM per Filon) = {delta_omega_for_filon:.4e} (rad/unità di tempo)"
    )

    filter_info_to_return = None
    if (
        gaussian_filter_width is not None and nmax_intervals > 0
    ):  # Filtro non ha senso per NMAX=0
        print(
            f"Applicazione del filtro Gaussiano con width_param = {gaussian_filter_width}..."
        )
        window_gauss = np.zeros(num_points_to_use)

        for i_point_idx in range(
            num_points_to_use
        ):  # i_point_idx da 0 a nmax_intervals
            # Fortran: i (1-based) da 1 a nmax (che è n_points_to_use nel mio codice Python)
            # DBLE(i-1) corrisponde a float(i_point_idx)
            # Il termine nel Fortran: (0.5d0 * width * DBLE(i-1)*dt / tmax)
            # Diventa: (0.5 * gaussian_filter_width * float(i_point_idx) * dt / t_max)
            #          = (0.5 * gaussian_filter_width * float(i_point_idx) / nmax_intervals)
            # Chiamiamo questo 'x_arg'
            x_arg_for_filter = (
                0.5 * gaussian_filter_width * float(i_point_idx) / float(nmax_intervals)
            )
            window_gauss[i_point_idx] = np.exp(-0.5 * x_arg_for_filter**2)
            current_corr_for_ft[i_point_idx] *= window_gauss[i_point_idx]

        print("  Finestra Gaussiana applicata.")
        filter_info_to_return = (
            time_points_to_use,
            np.copy(current_corr_for_ft),
            window_gauss,
        )
    elif gaussian_filter_width is not None and nmax_intervals == 0:
        print("  Filtro Gaussiano non applicato: NMAX=0 (un solo punto dati).")

    spectrum_chat_array = np.zeros(num_points_to_use)  # Lunghezza NMAX_intervals + 1

    _filon_cosine_transform_subroutine(
        dt_val=dt,
        delta_omega_val=delta_omega_for_filon,
        nmax_intervals=nmax_intervals,
        c_corr_func=current_corr_for_ft,
        chat_spectrum=spectrum_chat_array,
    )

    # Conversione finale dell'asse delle frequenze
    # Il 'dom' iniziale del Fortran era delta_omega_for_filon = 1.0 / t_max
    # L'asse delle frequenze per l'output è i * dom_converted
    # dom_converted = dom_initial / TWO_PI * HZ_TO_CMINV_FACTOR
    if output_in_wavenumbers:
        if t_max > 0:  # Evita divisione per zero se t_max è 0 (NMAX=0)
            # Frequenza lineare step: (Delta Omega) / 2pi
            linear_freq_step = delta_omega_for_filon / TWO_PI
            # Wavenumber step
            wavenumber_step = linear_freq_step * HZ_TO_CMINV_FACTOR
        else:  # NMAX=0, t_max=0. Solo la componente DC (nu=0) è significativa.
            # Per evitare NaN/Inf, impostiamo lo step a 0 ma l'asse avrà solo il punto a 0.
            wavenumber_step = 0.0

        frequencies_axis_final = np.arange(num_points_to_use) * wavenumber_step
        print(
            f"  Asse delle frequenze convertito in cm^-1 (step: {wavenumber_step:.4e} cm^-1)."
        )
    else:
        # Restituisci frequenze angolari omega = nu * delta_omega_for_filon
        frequencies_axis_final = np.arange(num_points_to_use) * delta_omega_for_filon
        print(
            f"  Asse delle frequenze in rad/unità_tempo (step: {delta_omega_for_filon:.4e})."
        )

        if norm:
            from scipy.integrate import simpson

            spectrum_chat_array /= simpson(
                y=spectrum_chat_array, x=frequencies_axis_final
            )

    return frequencies_axis_final, spectrum_chat_array, filter_info_to_return
