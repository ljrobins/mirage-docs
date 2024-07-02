"""
Loading with MTL Properties
===========================
"""

import os

import numpy as np

import mirage as mr

obj_name = 'untitled.obj'
obj_path = os.path.join('/Users/liamrobinson/Downloads/', obj_name)

# %%
# Let's take a look at what the obj file looks like

with open(obj_path, 'r') as f:
    print(f.read())

# %%
# And the mtl file
with open(os.path.join(os.path.split(obj_path)[0], 'untitled.mtl'), 'r') as f:
    print(f.read())

# %%
# We interpret:
#
# - The red channel of Kd (in Blender this is the red channel of the base color) as :math:`C_d`
#
# - The blue channel of Kd as :math:`C_s`
#
# - The index of refraction Ni (IOR in Blender) as the specular exponent :math:`n`
#
# For more information on making an mesh in blender with per-face materials, see `this documentation page <https://docs.blender.org/manual/en/4.1/modeling/texts/editing.html#assigning-materials>`_

mr.tic('Pure python load time')
obj = mr.load_obj(obj_path)
mr.toc()

# %%
# We can print the the cd, cs, and n attributes of the object, each of which should now have one entry per face

print(f'{obj.cd=}')
print(f'{obj.cs=}')
print(f'{obj.n=}')

# %%
# Let's build a BRDF with these attributes. Note if validate=True, the BRDF initialization procedure will check for energy conservation

brdf = mr.Brdf('phong', cd=obj.cd, cs=obj.cs, n=obj.n, validate=False)

# %%
# Now the BRDF will apply the material properties of each face when computing a convex LC

npts = int(100)
L = mr.rand_unit_vectors(npts)
O = mr.rand_unit_vectors(npts)
mr.tic('lc')
lc1 = obj.convex_light_curve(brdf, L, O)
mr.toc()

# %%
# Notice that you can also mix and match uniform and varying properties. Here :math:`C_d` has one entry per face, but the other two properties are uniform

brdf.cd = np.tile(brdf.cd[[0]], brdf.cd.shape)
brdf.cs = 0.9
brdf.n = 1
lc2 = obj.convex_light_curve(brdf, L, O)
