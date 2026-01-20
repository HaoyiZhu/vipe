import numpy as np
import torch
import pypose as pp
from scipy.spatial.transform import Rotation as R
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import time
import os
from typing import List, Tuple

# ==========================================
# Part 1: Solver (from fastloop/solve_python.py)
# ==========================================

def solve_sparse(A: csc_matrix, b: np.ndarray, freen: int) -> np.ndarray:
    """Solve linear system A * delta = b, supports submatrix solving"""
    if freen < 0:
        return spsolve(A, b)
    else:
        A_sub = A[:freen, :freen].tocsc()
        b_sub = b[:freen]
        delta_sub = spsolve(A_sub, b_sub)
        delta = np.zeros_like(b)
        delta[:freen] = delta_sub
        return delta

def solve_system_py(
    J_Ginv_i: torch.Tensor,
    J_Ginv_j: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    res: torch.Tensor,
    ep: float,
    lm: float,
    freen: int
) -> torch.Tensor:
    # Ensure all tensors are on CPU
    device = res.device
    J_Ginv_i = J_Ginv_i.cpu()
    J_Ginv_j = J_Ginv_j.cpu()
    ii = ii.cpu()
    jj = jj.cpu()
    res = res.clone().cpu()
    
    r = res.size(0)  # Number of edges
    n = max(ii.max().item(), jj.max().item()) + 1  # Number of nodes
    
    res_vec = res.view(-1).numpy().astype(np.float64)
    
    rows, cols, data = [], [], []
    ii_np = ii.numpy()
    jj_np = jj.numpy()
    J_Ginv_i_np = J_Ginv_i.numpy()
    J_Ginv_j_np = J_Ginv_j.numpy()
    
    # We can vectorize this loop for performance if needed, but sticking to original python implementation for exactness first.
    for x in range(r):
        i = ii_np[x]
        j = jj_np[x]
        if i == j:
            # Self-edges might occur if not careful, but usually shouldn't in this graph structure
            continue
        
        # J_i block (7x7)
        # J_j block (7x7)
        # 49 elements each
        
        # Flattened indices calculation
        # row_idx: x * 7 + k
        # col_idx_i: i * 7 + l
        
        row_indices = np.arange(x * 7, (x + 1) * 7).reshape(-1, 1).repeat(7, axis=1).flatten()
        
        col_indices_i = np.arange(i * 7, (i + 1) * 7).reshape(1, -1).repeat(7, axis=0).flatten()
        data_i = J_Ginv_i_np[x].flatten()
        rows.extend(row_indices)
        cols.extend(col_indices_i)
        data.extend(data_i)
        
        col_indices_j = np.arange(j * 7, (j + 1) * 7).reshape(1, -1).repeat(7, axis=0).flatten()
        data_j = J_Ginv_j_np[x].flatten()
        rows.extend(row_indices)
        cols.extend(col_indices_j)
        data.extend(data_j)
    
    J = coo_matrix((data, (rows, cols)), shape=(r * 7, n * 7)).tocsc()
    
    b_vec = - J.T @ res_vec
    
    A_mat = J.T @ J
    
    diag = A_mat.diagonal()
    new_diag = diag * (1.0 + lm) + ep
    A_mat.setdiag(new_diag)
    
    freen_total = freen * 7
    delta = solve_sparse(A_mat.tocsc(), b_vec, freen_total)
    
    delta_tensor = torch.from_numpy(delta.astype(np.float32)).view(n, 7).to(device)
    return delta_tensor


# ==========================================
# Part 2: Optimizer (from loop_utils/sim3loop.py)
# ==========================================

class Sim3LoopOptimizer:
    """
    Loop closure optimizer for sequences of Sim3 transformations using PyPose.
    """
    
    def __init__(self, device='cuda', max_iterations=30, lambda_init=1e-6):
        self.device = device
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
    
    def numpy_to_pypose_sim3(self, s: float, R_mat: np.ndarray, t_vec: np.ndarray) -> pp.Sim3:
        """Convert numpy s,R,t to pypose Sim3"""
        if isinstance(R_mat, torch.Tensor):
            R_mat = R_mat.cpu().numpy()
        if isinstance(t_vec, torch.Tensor):
            t_vec = t_vec.cpu().numpy()
        if isinstance(s, torch.Tensor):
            s = s.item()
            
        q = R.from_matrix(R_mat).as_quat()  # [x,y,z,w]
        # pypose requires [t, q, s] format for Sim3? 
        # Checking pypose docs or vggt implementation:
        # vggt impl: data = np.concatenate([t_vec, q, np.array([s])]) -> 3 + 4 + 1 = 8.
        # This assumes pypose Sim3 layout is [tx, ty, tz, qx, qy, qz, qw, s].
        data = np.concatenate([t_vec, q, np.array([s])])
        return pp.Sim3(torch.from_numpy(data).float().to(self.device))
    
    def pypose_sim3_to_numpy(self, sim3: pp.Sim3) -> Tuple[float, np.ndarray, np.ndarray]:
        """Convert pypose Sim3 to numpy s,R,t"""
        data = sim3.data.cpu().numpy()
        t = data[:3]
        q = data[3:7]  # [x,y,z,w]
        s = data[7]
        R_mat = R.from_quat(q).as_matrix()
        return s, R_mat, t
    
    def sequential_to_absolute_poses(self, sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]) -> torch.Tensor:
        """
        Convert sequential relative transforms to absolute pose sequence
        """
        poses = []
        # Identity in PyPose Sim3: [0,0,0, 0,0,0,1, 1] (t=0, q=[0,0,0,1], s=1)
        identity = pp.Sim3(torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], device=self.device))
        poses.append(identity)
        
        current_pose = identity
        for s, R_mat, t_vec in sequential_transforms:
            rel_transform = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            # vggt accumulates as current @ rel. 
            # In pipeline, sim3_list stores (s, R, t) for aligning i+1 to i. (T_{i+1->i})
            # P_{i+1} = P_i * T_{i+1->i} ?? 
            # If P_i is T_{i->w}, then P_{i+1} = P_i * T_{i+1->i} is NOT T_{i+1->w}.
            # It should be P_i * T_{i->i+1} = T_{i->w} * T_{w->i} * T_{i->i+1} ??? No.
            
            # Re-verify vggt assumption.
            # vggt code: `cumulative_transforms` logic matches `current @ rel`.
            # And `sim3_list` stores alignment result.
            # If the pipeline produces `weighted_align_point_maps(pt1, ..., pt2)` where pt1 is i, pt2 is i+1.
            # Then result is T_{i+1 -> i}.
            # If we assume P_i maps i to world. P_{i+1} = P_i * T_{i+1->i} makes P_{i+1} map (i+1) to world?
            # x_i = T_{i+1->i} * x_{i+1}.
            # x_w = P_i * x_i = P_i * T_{i+1->i} * x_{i+1}.
            # So P_{i+1} = P_i * T_{i+1->i} IS CORRECT for T_{i+1->w}.
            # Yes.
            
            current_pose = current_pose @ rel_transform
            poses.append(current_pose)
        
        return torch.stack(poses)
    
    def absolute_to_sequential_transforms(self, absolute_poses: pp.Sim3) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        sequential_transforms = []
        n = absolute_poses.shape[0]
        
        for i in range(n - 1):
            rel_transform = absolute_poses[i].Inv() @ absolute_poses[i + 1]
            s, R_mat, t_vec = self.pypose_sim3_to_numpy(rel_transform)
            sequential_transforms.append((s, R_mat, t_vec))
        
        return sequential_transforms
    
    def build_loop_constraints(self, 
                             loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not loop_constraints:
            return torch.empty(0, 8, device=self.device), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        loop_transforms = []
        ii_loop = []
        jj_loop = []
        
        for i, j, (s, R_mat, t_vec) in loop_constraints:
            loop_sim3 = self.numpy_to_pypose_sim3(s, R_mat, t_vec)
            loop_transforms.append(loop_sim3.data)
            ii_loop.append(i)
            jj_loop.append(j)
        
        if len(loop_transforms) > 0:
            dSloop = pp.Sim3(torch.stack(loop_transforms))
        else:
            dSloop = pp.Sim3(torch.empty(0, 8, device=self.device))
            
        ii_loop = torch.tensor(ii_loop, dtype=torch.long, device=self.device)
        jj_loop = torch.tensor(jj_loop, dtype=torch.long, device=self.device)
        
        return dSloop, ii_loop, jj_loop
    
    def residual(self, Ginv, input_poses, dSloop, ii, jj, jacobian=False):
        """Compute residuals (modified from original code)"""
        def _residual(C, Gi, Gj):
            # C @ Exp(Gi) @ Exp(Gj).Inv()
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()
        
        pred_inv_poses = pp.Sim3(input_poses).Inv()
        
        n, _ = pred_inv_poses.shape
        if n > 1:
            kk = torch.arange(1, n, device=self.device)
            ll = kk - 1
            Ti = pred_inv_poses[kk]
            Tj = pred_inv_poses[ll]
            dSij = Tj @ Ti.Inv() # P_{i-1}^{-1} * P_i = T_{i->i-1}
        else:
            kk = torch.empty(0, dtype=torch.long, device=self.device)
            ll = torch.empty(0, dtype=torch.long, device=self.device)
            dSij = pp.Sim3(torch.empty(0, 8, device=self.device))
        
        # Ensure ii and jj are on the correct device
        ii = ii.to(self.device)
        jj = jj.to(self.device)
        
        if dSloop.shape[0] > 0:
            constants = pp.Sim3(torch.cat((dSij.data, dSloop.data), dim=0))
            iii = torch.cat((kk, ii))
            jjj = torch.cat((ll, jj))
        else:
            constants = dSij
            iii = kk
            jjj = ll

        if constants.shape[0] > 0:
            resid = _residual(constants, Ginv[iii], Ginv[jjj])
        else:
            resid = torch.empty(0, device=self.device)

        if not jacobian:
            return resid

        if constants.shape[0] > 0:
            def batch_jacobian(func, x):
                def _func_sum(*x):
                    return func(*x).sum(dim=0)
                _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
                from einops import rearrange
                return rearrange(torch.stack((b, c)), 'N O B I -> N B O I', N=2)

            J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
        else:
            J_Ginv_i = torch.empty(0, device=self.device)
            J_Ginv_j = torch.empty(0, device=self.device)

        return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)
    
    def optimize(self, 
                sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]],
                loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]]) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        
        if not loop_constraints:
            # Match VGGT-Long: no loop constraints means no optimization.
            return sequential_transforms
        
        max_iterations = self.max_iterations
        lambda_init = self.lambda_init

        input_poses = self.sequential_to_absolute_poses(sequential_transforms)
        
        dSloop, ii_loop, jj_loop = self.build_loop_constraints(loop_constraints)
        
        # Ginv: Log of Inverse Poses
        # We optimize in tangent space of the inverse poses.
        # Note: vggt uses Ginv as state variable.
        Ginv = pp.Sim3(input_poses).Inv().Log()
        
        lmbda = lambda_init
        residual_history = []
        
        print(f"Starting Sim3 optimization with {len(sequential_transforms)} poses and {len(loop_constraints)} loops")
        
        for itr in range(max_iterations):
            resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual(
                Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)
            
            if resid.numel() == 0:
                break
                
            current_cost = resid.square().mean().item()
            residual_history.append(current_cost)
            
            try:
                # Use python solver
                delta_pose = solve_system_py(
                    J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1)
            except Exception as e:
                print(f"Solver failed at iteration {itr}: {e}")
                break
            
            Ginv_tmp = Ginv + delta_pose
            
            new_resid = self.residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
            new_cost = new_resid.square().mean().item() if new_resid.numel() > 0 else float('inf')
            
            # Levenberg-Marquardt Step
            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
                print(f"Iter {itr}: {current_cost:.6f} -> {new_cost:.6f} (accepted)")
            else:
                lmbda *= 2
                print(f"Iter {itr}: {current_cost:.6f} -> {new_cost:.6f} (rejected)")
            
            if (current_cost < 1e-5) and (itr >= 4):
                if len(residual_history) >= 5:
                    improvement_ratio = residual_history[-5] / (residual_history[-1] + 1e-12)
                    if improvement_ratio < 1.01:
                        print(f"Converged at iteration {itr}")
                        break
        
        optimized_absolute_poses = pp.Exp(Ginv).Inv()
        
        return self.absolute_to_sequential_transforms(optimized_absolute_poses)


# ==========================================
# Part 3: Utils (Alignment) (from loop_utils/sim3utils.py)
# ==========================================

def weighted_estimate_sim3(source_points, target_points, weights):
    # Same as before, but ensure clean implementation
    total_weight = torch.sum(weights)
    if total_weight < 1e-6:
        device = source_points.device
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    
    normalized_weights = weights / total_weight

    mu_src = torch.sum(normalized_weights[:, None] * source_points, dim=0)
    mu_tgt = torch.sum(normalized_weights[:, None] * target_points, dim=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    # Weighted Variance
    scale_src = torch.sqrt(torch.sum(normalized_weights * torch.sum(src_centered**2, dim=1)))
    scale_tgt = torch.sqrt(torch.sum(normalized_weights * torch.sum(tgt_centered**2, dim=1)))
    s = scale_tgt / (scale_src + 1e-12)

    weighted_src = (s * src_centered) * torch.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * torch.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, V = torch.svd(H) # torch.svd: H = U S V.T
    R = V @ U.T
    
    if torch.det(R) < 0:
        V_mod = V.clone()
        V_mod[:, 2] *= -1
        R = V_mod @ U.T

    t = mu_tgt - s * (R @ mu_src)
    return s.item(), R, t

def weighted_estimate_se3(source_points, target_points, weights):
    # Scale = 1.0
    total_weight = torch.sum(weights)
    if total_weight < 1e-6:
        device = source_points.device
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    
    normalized_weights = weights / total_weight

    mu_src = torch.sum(normalized_weights[:, None] * source_points, dim=0)
    mu_tgt = torch.sum(normalized_weights[:, None] * target_points, dim=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    weighted_src = src_centered * torch.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * torch.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, V = torch.svd(H)
    R = V @ U.T
    
    if torch.det(R) < 0:
        V_mod = V.clone()
        V_mod[:, 2] *= -1
        R = V_mod @ U.T

    t = mu_tgt - (R @ mu_src)
    
    return 1.0, R, t

def robust_weighted_align_point_maps(
    points1: torch.Tensor,
    conf1: torch.Tensor,
    points2: torch.Tensor,
    conf2: torch.Tensor,
    mask: torch.Tensor | None = None,
    conf_threshold: float = -1.0,
    delta: float = 0.1,
    max_iters: int = 5,
    tol: float = 1e-9,
    using_sim3: bool = True
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    def _as_tensor(x: torch.Tensor | np.ndarray | None, device: torch.device) -> torch.Tensor | None:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.from_numpy(np.asarray(x)).to(device)
    
    # Infer device and move all tensors there
    if isinstance(points1, torch.Tensor):
        device = points1.device
    elif isinstance(points2, torch.Tensor):
        device = points2.device
    else:
        device = torch.device("cpu")
    
    points1 = _as_tensor(points1, device).float()
    points2 = _as_tensor(points2, device).float()
    conf1 = _as_tensor(conf1, device).float()
    conf2 = _as_tensor(conf2, device).float()
    mask = _as_tensor(mask, device)
    if mask is not None:
        mask = mask.bool()
    
    # Support (N,3), (B,N,3) and (B,H,W,3) formats
    if points1.ndim == 2:
        if conf1.shape != conf2.shape:
            return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
        weights = torch.sqrt(conf1 * conf2)
        valid_mask = torch.ones_like(weights, dtype=torch.bool)
        if conf_threshold > 0:
            valid_mask &= (conf1 > conf_threshold)
            valid_mask &= (conf2 > conf_threshold)
        if mask is not None and mask.shape == valid_mask.shape:
            valid_mask &= mask
        
        pts1 = points1[valid_mask]  # Target
        pts2 = points2[valid_mask]  # Source
        init_weights = weights[valid_mask]
    else:
        b = min(points1.shape[0], points2.shape[0])
        points1 = points1[:b]
        points2 = points2[:b]
        conf1 = conf1[:b]
        conf2 = conf2[:b]
        if mask is not None:
            mask = mask[:b]
        
        pts1_list = []
        pts2_list = []
        weights_list = []
        
        for i in range(b):
            if points1.ndim == 4:
                conf1_i = conf1[i].squeeze()
                conf2_i = conf2[i].squeeze()
                if conf1_i.shape != conf2_i.shape:
                    continue
                valid = torch.ones_like(conf1_i, dtype=torch.bool)
                
                if conf_threshold > 0:
                    valid &= (conf1_i > conf_threshold)
                    valid &= (conf2_i > conf_threshold)
                
                if mask is not None:
                    mask_i = mask[i].squeeze()
                    if mask_i.shape == valid.shape:
                        valid &= mask_i
                
                idx = valid.nonzero(as_tuple=True)
                if idx[0].numel() == 0:
                    continue
                
                pts1_i = points1[i][idx]
                pts2_i = points2[i][idx]
                w_i = torch.sqrt(conf1_i[idx] * conf2_i[idx])
            else:
                conf1_i = conf1[i].squeeze()
                conf2_i = conf2[i].squeeze()
                if points1[i].shape[0] != points2[i].shape[0]:
                    continue
                if conf1_i.shape != conf2_i.shape:
                    continue
                valid = torch.ones_like(conf1_i, dtype=torch.bool)
                
                if conf_threshold > 0:
                    valid &= (conf1_i > conf_threshold)
                    valid &= (conf2_i > conf_threshold)
                
                if mask is not None:
                    mask_i = mask[i].squeeze()
                    if mask_i.shape == valid.shape:
                        valid &= mask_i
                
                pts1_i = points1[i][valid]
                pts2_i = points2[i][valid]
                w_i = torch.sqrt(conf1_i[valid] * conf2_i[valid])
            
            if pts1_i.numel() == 0:
                continue
            
            pts1_list.append(pts1_i)
            pts2_list.append(pts2_i)
            weights_list.append(w_i)
        
        if not pts1_list:
            return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
        
        pts1 = torch.cat(pts1_list, dim=0)
        pts2 = torch.cat(pts2_list, dim=0)
        init_weights = torch.cat(weights_list, dim=0)
    
    if pts1.shape[0] < 3:
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    
    src = pts2
    tgt = pts1
    
    if using_sim3:
        s, R, t = weighted_estimate_sim3(src, tgt, init_weights)
    else:
        s, R, t = weighted_estimate_se3(src, tgt, init_weights)
        
    prev_error = float('inf')
    
    for _ in range(max_iters):
        transformed = s * (src @ R.T) + t
        residuals = torch.norm(tgt - transformed, dim=1)
        
        abs_r = residuals
        huber_w = torch.ones_like(residuals)
        mask_large = abs_r > delta
        huber_w[mask_large] = delta / abs_r[mask_large]
        
        combined_weights = init_weights * huber_w
        
        if using_sim3:
            s_new, R_new, t_new = weighted_estimate_sim3(src, tgt, combined_weights)
        else:
            s_new, R_new, t_new = weighted_estimate_se3(src, tgt, combined_weights)
            
        param_change = abs(s_new - s) + torch.norm(t_new - t).item()
        
        # Rotation change check
        trace = torch.trace(R_new @ R.T)
        rot_angle = torch.acos(torch.clamp((trace - 1)/2, -1.0, 1.0)).item()
        
        # Huber Loss
        huber_loss = torch.where(abs_r <= delta, 0.5 * abs_r**2, delta * (abs_r - 0.5 * delta))
        current_error = (huber_loss * init_weights).sum().item()
        
        if (param_change < tol and rot_angle < 0.0017) or \
           (abs(prev_error - current_error) < tol * prev_error):
            s, R, t = s_new, R_new, t_new
            break
            
        s, R, t = s_new, R_new, t_new
        prev_error = current_error
        
    return s, R, t

def compute_sim3_ab(sim3_a, sim3_b):
    s_a, R_a, t_a = sim3_a
    s_b, R_b, t_b = sim3_b
    
    if any(isinstance(x, torch.Tensor) for x in (R_a, R_b, t_a, t_b)):
        device = None
        for x in (R_a, R_b, t_a, t_b):
            if isinstance(x, torch.Tensor):
                device = x.device
                break
        device = device or torch.device("cpu")
        
        R_a_t = R_a if isinstance(R_a, torch.Tensor) else torch.tensor(R_a, device=device)
        R_b_t = R_b if isinstance(R_b, torch.Tensor) else torch.tensor(R_b, device=device)
        R_a_t = R_a_t.to(device)
        R_b_t = R_b_t.to(device)
        t_a_t = t_a if isinstance(t_a, torch.Tensor) else torch.tensor(t_a, device=device)
        t_b_t = t_b if isinstance(t_b, torch.Tensor) else torch.tensor(t_b, device=device)
        s_a_t = s_a if isinstance(s_a, torch.Tensor) else torch.tensor(s_a, device=device)
        s_b_t = s_b if isinstance(s_b, torch.Tensor) else torch.tensor(s_b, device=device)
        
        s_ab = s_b_t / s_a_t
        R_ab = R_b_t @ R_a_t.transpose(-1, -2)
        t_ab = t_b_t - s_ab * (R_ab @ t_a_t)
        return s_ab, R_ab, t_ab
    
    s_ab = s_b / s_a
    R_ab = R_b @ R_a.T
    t_ab = t_b - s_ab * (R_ab @ t_a)
    
    return s_ab, R_ab, t_ab
