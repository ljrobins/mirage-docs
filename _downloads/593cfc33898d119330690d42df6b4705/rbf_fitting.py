"""
Neural Network Brightness
=========================

Trains a neural network to predict the brightness of a specular cube in an arbitrary lighting and observation conditions and compares the results to the truth
"""

import sys

sys.path.append("./src")
import numpy as np
import pyspaceaware as ps
import pyspaceaware.sim as pssim
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Let's define the object and the BRDF
obj = ps.SpaceObject("cube.obj")
brdf = ps.Brdf("phong", cd=0.5, cs=0.5, n=10)
# %%
# We now define the Multi-Layer Perceptron (MLP) brightness model. Note that the ``layers=(150, 50, 150)`` keyword argument defines the number of neurons in each densely-connected layer.
mlp_bm = pssim.MLPBrightnessModel(
    obj, brdf, use_engine=False, train_on="magnitude"
)
# %%
# Now we train the model on a set number of training lighting and observation configurations. Usually ``1e5``-``1e6`` are required for a *good* fit
num_train = int(1e3)
mlp_bm.train(num_train)

# %%
# We can now simulate a torque-free attitude profile to inspect the quality of the fit
t_eval = np.linspace(0, 10, 1000)
q, _ = ps.propagate_attitude_torque_free(
    np.array([0.0, 0.0, 0.0, 1.0]),
    np.array([1.0, 1.0, 1.0]),
    np.diag([1, 2, 3]),
    t_eval,
)
dcm = ps.quat_to_dcm(q)
ovb = ps.stack_mat_mult(dcm, np.array([[1, 0, 0]]))
svb = ps.stack_mat_mult(dcm, np.array([[0, 1, 0]]))

# %%
# Evaluating the model in its two available formats - as a native ``scikit-learn`` model and as an Open Neural Network eXchange (ONNX) model
ps.tic("Evaluate trained model with sklearn")
mdl_b_sklearn = mlp_bm.eval(ovb, svb, eval_mode_pref="sklearn")
ps.toc()
ps.tic("Evaluate trained model with onnx")
mdl_b_onnx = mlp_bm.eval(ovb, svb, eval_mode_pref="onnx")
ps.toc()

# %%
# We can save both of these representations to file:
mlp_bm.save_to_file(save_as_format="onnx")
mlp_bm.save_to_file(save_as_format="sklearn")

# %%
# Now we load the model from its ``.onxx`` file we just saved and evaluate the brightness
mlp_bm.load_from_file(mlp_bm.onnx_file_name)
ps.tic("Evaluate loaded model with onxx")
mdl_onnx_loaded = mlp_bm.eval(ovb, svb, eval_mode_pref="onnx")
ps.toc()

# %%
# And we do the same for the ``scikit-learn`` ``.plk`` file we saved
mlp_bm.load_from_file(mlp_bm.sklearn_file_name)
ps.tic("Evaluate loaded model with sklearn")
mdl_sklearn_loaded = mlp_bm.eval(ovb, svb, eval_mode_pref="sklearn")
ps.toc()

# %%
# We can easily confirm that all four model evaluations have produced the same prediction
print(np.max(np.abs(mdl_b_sklearn - mdl_onnx_loaded)))
print(np.max(np.abs(mdl_b_onnx - mdl_onnx_loaded)))
print(np.max(np.abs(mdl_b_sklearn - mdl_sklearn_loaded)))
print(np.max(np.abs(mdl_b_onnx - mdl_sklearn_loaded)))

# %%
# We can now finish off by evaluating the true brightness in this attitude profile and plot the results
true_b = mlp_bm.brightness(svb, ovb, "magnitude")

plt.figure()
sns.lineplot(x=t_eval, y=true_b, errorbar=None)
sns.lineplot(x=t_eval, y=mdl_b_sklearn, errorbar=None)
plt.title(
    f"Light Curves for {obj.file_name}, {num_train} Training Points"
)
plt.xlabel("Time [s]")
plt.ylabel("Normalized brightness")
plt.legend(["True", "Model"])
plt.grid()
plt.show()
