import argparse
import time
import numpy as np
import viser
import viser.transforms as tf

def main():
    parser = argparse.ArgumentParser(description="Visualize camera trajectory from N*4*4 poses")
    parser.add_argument("pose_file", type=str, help="Path to numpy file containing N*4*4 poses")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--frustum-scale", type=float, default=0.1, help="Scale of camera frustums")
    parser.add_argument("--axes-len", type=float, default=0.05, help="Length of camera axes")
    parser.add_argument("--pose-file2", type=str, default=None, help="Path to second numpy file containing N*4*4 poses (comparison)")
    parser.add_argument("--fov", type=float, default=60.0, help="Field of view in degrees")
    parser.add_argument("--connect", action="store_true", default=True, help="Draw a spline connecting the cameras")
    args = parser.parse_args()

    print(f"Loading poses from {args.pose_file}...")
    try:
        poses = np.load(args.pose_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        print(f"Error: Expected shape (N, 4, 4), got {poses.shape}")
        return
    
    print(f"Loaded {len(poses)} poses.")

    # Optional second trajectory
    poses2 = None
    if args.pose_file2:
        print(f"Loading comparison poses from {args.pose_file2}...")
        try:
            poses2 = np.load(args.pose_file2)
        except Exception as e:
            print(f"Error loading comparison file: {e}")
            return
        
        if poses2.ndim != 3 or poses2.shape[1:] != (4, 4):
            print(f"Error: Comparison poses shape mismatch: {poses2.shape}")
            return
            
        if len(poses2) != len(poses):
            print(f"Warning: Trajectory lengths differ ({len(poses)} vs {len(poses2)}). Alignment might be poor if not synchronized.")
            # We will proceed by truncating to the minimum length for alignment, 
            # but visualize all.
            
        # Alignment (Umeyama)
        # 1. Extract centers
        # We align P2 to P1 (Reference)
        
        # Use common length for alignment calculation
        n_align = min(len(poses), len(poses2))
        p1 = poses[:n_align, :3, 3]
        p2 = poses2[:n_align, :3, 3]
        
        # Center
        mu1 = p1.mean(axis=0)
        mu2 = p2.mean(axis=0)
        
        p1_c = p1 - mu1
        p2_c = p2 - mu2
        
        # Covariance
        H = p2_c.T @ p1_c
        U, S, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T
        
        # Reflection check
        if np.linalg.det(R_align) < 0:
            Vt[2, :] *= -1
            R_align = Vt.T @ U.T
            
        # Scale
        var1 = np.sum(p1_c**2)
        var2 = np.sum(p2_c**2)
        scale = np.sqrt(var1 / var2)
        
        # Translation
        t_align = mu1 - scale * (R_align @ mu2)
        
        print(f"Alignment: Scale={scale:.4f}, Translation={t_align}")
        
        # Apply transform to poses2
        # T_new = [ R_align * R_old   scale * R_align * t_old + t_align ]
        #         [ 0                 1                                 ]
        
        poses2_aligned = []
        for c2w in poses2:
            R_old = c2w[:3, :3]
            t_old = c2w[:3, 3]
            
            R_new = R_align @ R_old
            t_new = scale * (R_align @ t_old) + t_align
            
            new_pose = np.eye(4)
            new_pose[:3, :3] = R_new
            new_pose[:3, 3] = t_new
            poses2_aligned.append(new_pose)
        
        poses2 = np.array(poses2_aligned)

    server = viser.ViserServer(port=args.port)
    
    # Function to draw a trajectory
    def draw_trajectory(traj_poses, prefix, base_color, is_reference=True):
        positions = []
        for i, c2w in enumerate(traj_poses):
            position = c2w[:3, 3]
            positions.append(position)
            
            # Determine color
            if i == 0:
                color = (0, 255, 0) # Green for start
                name_suffix = " (Start)"
                scale_multiplier = 1.5
            elif i == len(traj_poses) - 1:
                color = (255, 0, 0) # Red for end
                name_suffix = " (End)"
                scale_multiplier = 1.5
            else:
                color = base_color
                name_suffix = ""
                scale_multiplier = 1.0

            # Create frame and frustum
            server.scene.add_frame(
                f"{prefix}/frames/t{i}",
                position=position,
                wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
                axes_length=args.axes_len * scale_multiplier,
                axes_radius=args.axes_len * 0.1 * scale_multiplier,
            )
            
            # Add frustum
            server.scene.add_camera_frustum(
                f"{prefix}/frames/t{i}/frustum",
                fov=np.deg2rad(args.fov),
                aspect=1.33,
                scale=args.frustum_scale * scale_multiplier,
                color=color,
            )
            
            # Add label for start/end
            if i == 0 or i == len(traj_poses) - 1:
                server.scene.add_label(
                    f"{prefix}/labels/t{i}",
                    text=f"{prefix} {i}{name_suffix}",
                    position=position,
                )

        if args.connect and len(positions) > 1:
            server.scene.add_spline_catmull_rom(
                f"{prefix}/trajectory",
                positions=np.array(positions),
                color=base_color,
                line_width=3.0,
                segments=len(positions) * 4
            )

    # Draw first trajectory
    draw_trajectory(poses, "/traj1", (100, 100, 255))
    
    # Draw second trajectory if exists
    if poses2 is not None:
        draw_trajectory(poses2, "/traj2_aligned", (255, 0, 255), is_reference=False)

    print(f"Viser server running at http://localhost:{args.port}")
    print("Press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
