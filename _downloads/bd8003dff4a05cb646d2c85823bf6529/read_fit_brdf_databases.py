"""
Fitting BRDF Databases
======================
"""

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.optimize import minimize

import mirage as mr

_DB_DIR = "/Users/liamrobinson/Documents/brdfmachine/BRDFDatabase"
_EXE_DIR = os.path.join(_DB_DIR, "code")

brdf_file = "orange-paint"
brdf_model = "cook-torrance"
brdf_path = os.path.join(_DB_DIR, "brdfs", f"{brdf_file}.binary")
cmd = f"cd {_EXE_DIR} && ./BRDFRead {brdf_path}"

mr.tic()
proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
brdf_info = proc.stdout
lines = brdf_info.strip().split("\n")
data = np.array([[float(value) for value in line.split()] for line in lines])
# theta_in, phi_in, theta_out, phi_out, red, green, blue
# theta angle from N, phi angle from tangent
mr.toc()

n = np.tile(np.array([[0.0, 0.0, 1.0]]), (data.shape[0], 1))
sv = mr.stack_mat_mult_vec(mr.r1(data[:, 0]), n)
n_to_ov_rotm = mr.stack_mat_mult_mat(mr.r3(data[:, 3] - data[:, 1]), mr.r1(data[:, 2]))
ov = mr.stack_mat_mult_vec(n_to_ov_rotm, n)

b_true = np.mean(data[:, -3:], axis=1).flatten()

print(b_true)

def eval_brdf_fit(x: np.ndarray) -> float:
    brdf = mr.Brdf(brdf_model, cd=x[0], cs=x[1], n=x[2], validate=False)
    b_est = brdf.eval(sv, ov, n).flatten()
    err = np.linalg.norm(b_true - b_est)
    print(err)
    return err


sol = minimize(eval_brdf_fit, x0=(0.5, 0.5, 10))

brdf_opt = mr.Brdf(brdf_model, cd=sol.x[0], cs=sol.x[1], n=sol.x[2], validate=False)
b_opt = brdf_opt.eval(sv, ov, n).flatten()
print(np.vstack((b_true, b_opt)).T)
print(np.max(np.abs(b_true - b_opt)))
print(brdf_opt)
