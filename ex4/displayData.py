#DISPLAYDATA Display 2D data in a nice grid
#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
#   stored in X in a nice grid. It returns the figure handle h and the
#   displayed array if requested.
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.misc
import matplotlib.cm as cm

def displayData(X):
	m = X.shape[0] # size 5000 or the number chosen
	n = X.shape[1] # size 400 = 20*20

	# Set size of single example
	example_width = int(math.sqrt(n)) # 20
	example_height = int(n/example_width) # 20
	#print("example_width:", example_width, "example_height:", example_height)

	# Compute number of items to display
	display_rows = int(math.sqrt(m))
	display_cols = math.ceil(m/display_rows)
	#print("display_rows:", display_rows, "display_cols:", display_cols)

	# Between images padding
	pad = 1;
	# Setup blank display
	rowsize = pad + display_rows * (example_height + pad)
	colsize = pad + display_cols * (example_width + pad)
	#print("rowsize:", rowsize, "colsize:", colsize)

	display_array = -np.ones((rowsize, colsize))

	# Copy each sample into a patch on the display array
	for curr_sample in range(m):
		# convert the example (in a single row) to a square
		squareX = X[curr_sample, :].reshape(example_height, example_width)
		squareX = squareX.T

		# find the position in the display_array
		rowpos = math.floor(curr_sample/display_cols)
		colpos = curr_sample % display_cols
		# get the single sample's start and end index
		rowstart = pad + rowpos * (example_height + pad)
		rowend = rowstart + example_height
		colstart = pad + colpos * (example_width + pad)
		colend = colstart + example_width
		#print("curr_sample", curr_sample, "rowstart", rowstart, "colstart", colstart)
		display_array[rowstart:rowend, colstart:colend] = squareX

	# Display Image
	fig = plt.figure(figsize=(display_cols/2,display_cols/2))
	img = scipy.misc.toimage( display_array )
	plt.imshow(img, cmap = "gray") #plt.imshow(img, cmap = cm.Greys_r)
	plt.axis('off')
