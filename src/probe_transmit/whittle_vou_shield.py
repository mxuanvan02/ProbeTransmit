class WhittleVoUShield(Policy):
    """Idea 2: Whittle/VoI ranking with VoU-style empirical violation shield.
    
    Two-tier candidate selection:
    1. Shield set (priority): loops with high empirical violation probability
    2. Whittle fill: remaining budget filled by descending Whittle VoI index
    
    This hybrid preserves Whittle's RMSE edge while importing VoU's winning
    safety mechanism (the source of its loss_mean victory).
    """
    
    name = "whittle_vou_shield"
    
    def __init__(self, lambda_safety: float = 15.0, shield_threshold: float = 0.3,
                 shield_budget_fraction: float = 0.5, deg: int = 5,
                 metadata_noise_std: float = 0.0):
        self.lambda_safety = float(lambda_safety)
        self.shield_threshold = float(shield_threshold)
        self.shield_budget_fraction = float(shield_budget_fraction)
        self.deg = int(deg)
        self.metadata_noise_var = float(metadata_noise_std) ** 2
        nodes, weights = np.polynomial.hermite.hermgauss(self.deg)
        self.gh_nodes = nodes
        self.gh_weights = weights
        self.gh_norm = 1.0 / float(np.sqrt(np.pi))
    
    def _payload_index_empirical(self, mu: np.ndarray, var: np.ndarray,
                                  xh: np.ndarray, p_succ: float,
                                  ar_model) -> np.ndarray:
        """Payload index using empirical safety probability."""
        var_safe = np.maximum(var, 1e-12)
        track_term = ((mu - xh) ** 2 + var_safe) / (safety.RANGE ** 2)
        
        # Use empirical tail probability
        p_violate = ar_model.empirical_safety_prob(
            mu, var_safe, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
        )
        
        xh_safe = ((xh >= safety.SAFE_MIN) & (xh <= safety.SAFE_MAX)).astype(float)
        return p_succ * (track_term + self.lambda_safety * p_violate * xh_safe)
    
    def select_probe(self, state: SchedulerState) -> np.ndarray:
        mu, var = state.ar.forecast_stats(state.xh, state.age, h=4)
        p_succ = float(np.clip(predict_success(state.pi_bad, state.channel), 0.25, 1.0))
        
        # Step 1: Identify shield set (high violation probability loops)
        pvio = state.ar.empirical_safety_prob(
            mu, var, safe_min=safety.SAFE_MIN, safe_max=safety.SAFE_MAX
        )
        shield_mask = pvio >= self.shield_threshold
        shield_indices = np.where(shield_mask)[0]
        
        # Step 2: Compute Whittle VoI index for all loops
        i2_prior = self._payload_index_empirical(mu, var, state.xh, p_succ, state.ar)
        
        # Adaptive Lagrange multiplier
        if state.b_payload <= 0:
            nu_2 = float(np.max(i2_prior)) + 1.0
        elif state.b_payload >= state.n:
            nu_2 = 0.0
        else:
            sorted_desc = np.sort(i2_prior)[::-1]
            nu_2 = float(sorted_desc[min(state.b_payload, len(sorted_desc)-1)])
        
        v_no_probe = np.maximum(i2_prior - nu_2, 0.0)
        
        # Gauss-Hermite quadrature for V_probe
        y_var = var + self.metadata_noise_var
        K = var / np.maximum(y_var, 1e-12)
        post_var = np.maximum((1.0 - K) * var, 0.0)
        y_sd = np.sqrt(np.maximum(y_var, 1e-12))
        sqrt2 = np.sqrt(2.0)
        
        v_probe = np.zeros(state.n, dtype=float)
        for k_idx in range(self.deg):
            z_k = float(self.gh_nodes[k_idx])
            w_k = float(self.gh_weights[k_idx])
            y_k = mu + z_k * sqrt2 * y_sd
            mu_post = mu + K * (y_k - mu)
            i2_post = self._payload_index_empirical(mu_post, post_var, state.xh, p_succ, state.ar)
            v_probe += w_k * np.maximum(i2_post - nu_2, 0.0)
        v_probe *= self.gh_norm
        
        whittle_index = v_probe - v_no_probe
        
        # Step 3: Two-tier selection
        # First, allocate shield budget to high-pvio loops
        max_shield = int(self.shield_budget_fraction * state.b_probe)
        if len(shield_indices) > 0:
            # Sort shield candidates by pvio (highest first)
            shield_pvio = pvio[shield_indices]
            shield_order = np.argsort(-shield_pvio)
            shield_selected = shield_indices[shield_order[:min(max_shield, len(shield_indices))]]
        else:
            shield_selected = np.array([], dtype=int)
        
        # Then fill remaining budget with Whittle ranking (excluding already selected)
        remaining_budget = state.b_probe - len(shield_selected)
        if remaining_budget > 0:
            # Mask out already selected
            whittle_masked = whittle_index.copy()
            whittle_masked[shield_selected] = -np.inf
            whittle_fill = topk(whittle_masked, remaining_budget)
        else:
            whittle_fill = np.array([], dtype=int)
        
        # Combine
        selected = np.concatenate([shield_selected, whittle_fill]).astype(int)
        return selected[:state.b_probe]
