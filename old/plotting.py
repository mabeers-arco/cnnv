import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import numpy as np



#########################################################################################
with open('./data/everything.pickle', 'rb') as handle:
    everything = pickle.load(handle)

#########################################################################################
# ###### PLOT: True slant angle vs. estimated slant angle   #####

# true_SAs = []
# est_SAs = []
# for k in range(len(everything['uid'])):
# 	if type(everything["true_slant_angle"][k]) != float and type(everything['loss4results'][k]) != float:
# 		if everything["true_slant_angle"][k]['fun'] <= 1e-10:
# 			true_SAs += [np.degrees(everything["true_slant_angle"][k]['x'][0])]
# 			est_SAs += [np.degrees(everything['loss4results'][k]['x'])]

# est_SAs = np.hstack(est_SAs)

# plt.scatter(true_SAs, est_SAs, alpha=.5)
# plt.xlabel("True Slant Angle")
# plt.ylabel("Estimated Slant Angle")
# plt.plot([0,90], [0,90], 'r--')
# plt.savefig("./plots/slant_angle.png")
# plt.show()

#########################################################################################
# ### PLOT: True slant angle vs. estimated slant angle AVERAGE  #####

# true_SAs = []
# est_SAs = []
# for k in range(len(everything['uid'])):
# 	if type(everything["true_slant_angle"][k]) != float and type(everything['loss4results'][k]) != float:
# 		if everything["true_slant_angle"][k]['fun'] <= 1e-10:
# 			true_SAs += [np.degrees(everything["true_slant_angle"][k]['x'][0])]
# 			est_SAs += [np.degrees(everything['loss4results'][k]['x'])]

# est_SAs = np.hstack(est_SAs)
# true_SAs = np.array(true_SAs)
# step_size = 5
# angles = np.arange(0, 90 + step_size, step_size)
# avg_x = []
# avg_y = []
# for k in range(len(angles) - 1):
# 	lower = angles[k]
# 	upper = angles[k+1]
# 	avg_x += [(lower + upper)/2]
# 	avg_y += [np.mean(est_SAs[(true_SAs >= lower) & (true_SAs < upper)])]

# avg_x = np.hstack(avg_x)
# avg_y = np.hstack(avg_y)


# plt.scatter(avg_x, avg_y, alpha=.5)
# plt.plot(avg_x, avg_y, alpha=.5)
# plt.plot([0,90], [0,90], '--')
# plt.xlabel("True Slant Angle")
# plt.ylabel("Estimated Slant Angle")
# plt.title("Averaged over {} degrees".format(step_size))
# plt.axis('equal')
# plt.savefig("./plots/slant_angle_avg{}.png".format(step_size))
# plt.show()

#########################################################################################
# ###### PLOT: Dissimilarity measure (y) defined in Y.Li, (2011) vs true slant angle (x) #####

# true_SAs = []
# dissimilarity = []
# for k in range(len(everything['uid'])):
# 	if type(everything["true_slant_angle"][k]) != float and type(everything['loss4results'][k]) != float:
# 		if everything["true_slant_angle"][k]['fun'] <= 1e-10:
# 			alpha1 = everything["true_slant_angle"][k]['x'][0]
# 			alpha2 = everything['loss4results'][k]['x']
# 			true_SAs += [np.degrees(alpha1)]
# 			en = np.cos(alpha1)/np.cos(alpha2)
# 			em = np.sin(alpha1)/np.sin(alpha2)
# 			d = np.log2(np.abs(en/em))
# 			dissimilarity += [d]




# dissimilarity = np.hstack(dissimilarity)

# plt.scatter(true_SAs, dissimilarity, alpha=.5)
# plt.xlabel("True Slant Angle")
# plt.ylabel("Dissimilarity")
# plt.savefig("./plots/dissimilarity.png")
# plt.show()


#########################################################################################
# ###### PLOT: Histograms of true vs estimated slant angle #####

# true_SAs = []
# est_SAs = []
# for k in range(len(everything['uid'])):
# 	if type(everything["true_slant_angle"][k]) != float and type(everything['loss4results'][k]) != float:
# 		true_SAs += [np.degrees(everything["true_slant_angle"][k]['x'][0])]
# 		est_SAs += [np.degrees(everything['loss4results'][k]['x'])]

# est_SAs = np.hstack(est_SAs)


# fig, (ax1, ax2) = plt.subplots(1,2, figsize = (11, 5), sharey=True)
# ax1.hist(true_SAs)
# ax1.set_xlabel("Angle (Degrees)")
# ax1.set_ylabel("Frequency")
# ax1.set_title("True Slant Angle")
# #ax1.set_ylim([0, 900])

# ax2.hist(est_SAs)
# ax2.set_xlabel("Angle (Degrees)")
# ax2.set_ylabel("Frequency")
# ax2.set_title("Estimated Slant Angle")

# plt.savefig("./plots/SA_histograms.png")
# plt.show()

#########################################################################################
# ###### PLOT: Dissimilarity measure (y) defined in Y.Li, (2011) vs true slant angle (x) AVERAGE ###

true_SAs = []
dissimilarity = []
for k in range(len(everything['uid'])):
	if type(everything["true_slant_angle"][k]) != float and type(everything['loss4results'][k]) != float:
		if everything["true_slant_angle"][k]['fun'] <= 1e-10:
			alpha1 = everything["true_slant_angle"][k]['x'][0]
			alpha2 = everything['loss4results'][k]['x']
			true_SAs += [np.degrees(alpha1)]
			en = np.cos(alpha1)/np.cos(alpha2)
			em = np.sin(alpha1)/np.sin(alpha2)
			d = np.log2(np.abs(en/em))
			dissimilarity += [d]


dissimilarity = np.hstack(dissimilarity)
true_SAs = np.array(true_SAs)
step_size = 10
angles = np.arange(0, 90 + step_size, step_size)
avg_x = []
avg_y = []
for k in range(len(angles) - 1):
	lower = angles[k]
	upper = angles[k+1]
	avg_x += [(lower + upper)/2]
	avg_y += [np.mean(dissimilarity[(true_SAs >= lower) & (true_SAs < upper)])]

avg_x = np.hstack(avg_x)
avg_y = np.hstack(avg_y)

plt.scatter(avg_x, avg_y, alpha=.5)
plt.plot(avg_x, avg_y, alpha=.5)
plt.xlabel("True Slant Angle")
plt.ylabel("Dissimilarity")
plt.savefig("./plots/dissimilarity_avg{}.png".format(step_size))
plt.show()

#########################################################################################
# #### PLOT: Add Noise to input 2D shape, see how recovered slant angle changes #######

# from scipy.optimize import minimize
# from optimize import loss4

# n = len(everything["uid"])
# n_objects = 10
# n_samples = 11
# n_sds = 6
# obj_inds = np.random.randint(0, n, n_objects)
# sds = np.linspace(.01, .2, n_sds)
# df = np.empty((len(sds), n_samples, n_objects))
# df[:] = np.NaN
# eps = 1e-12
# bounds_theta = [[0 + eps, np.pi/2 - eps]]


# for i, sd in enumerate(sds):
# 	for j, idx in enumerate(obj_inds):
# 		xyz = everything["xyz_rotated"][idx]
# 		if type(xyz) == float:
# 			continue
# 		mpl = everything["mpl"][idx]
# 		faces = everything["faces"][idx]
# 		xy = xyz[:, :2]
# 		if len(mpl) == 3:
# 			for k in range(n_samples):
# 				noise2d = sd * np.random.randn(xy.shape[0], xy.shape[1])
# 				xy_new = xy + noise2d
# 				opt_results = minimize(lambda alpha: loss4(alpha, xy_new, faces, mpl), 
# 					x0 = 1, 
# 					bounds = bounds_theta)
# 				sa = opt_results['x']
# 				df[i,k,j] = np.degrees(sa)

# 	print("sd = {} completed".format(sd))


# ## df dimensions: sd[i] x sample[k] x object[j]
# ## care about deviation from true slant angle

# DEVS = []
# for i in range(n_sds):
# 	deviations = []
# 	for k, idx in enumerate(obj_inds):
# 		true_sa = everything['loss4results'][idx]
# 		if not np.isnan(df[i, 0, k]) and type(true_sa) is not float: 
# 			true_sa = np.degrees(true_sa['x'].item())
# 			d = df[i, :, k] - true_sa
# 			deviations += [d]
# 	deviations = np.hstack(deviations)
# 	DEVS += [deviations]



# fig, (ax1, ax2) = plt.subplots(1,2, figsize = (11, 5))

# # Real Data
# x_devs = []
# for i, dev in enumerate(DEVS):
# 	x_devs += [sds[i]]*len(dev)
# ax1.scatter(x_devs , np.hstack(DEVS), alpha = .5)
# ax1.set_xlabel("SD of Noise Added to 2D input")
# ax1.set_ylabel("Recovered slant angle deviations")

# #standard deviations
# stds = np.hstack([np.std(d) for d in DEVS])
# ax2.plot(sds, stds)
# ax2.scatter(sds, stds)
# ax2.set_xlabel("SD of Noise Added to 2D input")
# ax2.set_ylabel("SD of recovered slant angle deviations")

# plt.suptitle("n objects = {}, sds = {}, n samples = {}".format(n_objects, np.round(sds, 3), n_samples))
# plt.savefig("./plots/noise2d.png")
# plt.show()


#########################################################################################






















