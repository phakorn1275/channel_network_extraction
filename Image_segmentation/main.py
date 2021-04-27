import core as c
import skimage.io as skio
import numpy as np


'''
keep in mind the true band order might not follow this sequence
0  B1	Aerosols			60 meters (bad)
1  B2	Blue			    10 meters (moderate)
2  B3	Green			    10 meters (moderate)
3  B4	Red			        10 meters (moderate)
4  B5	Red Edge 1			20 meters (moderate)
5  B6	Red Edge 2			20 meters (good)
6  B7	Red Edge 3			20 meters (good)
7  B8	NIR			        10 meters (good)
8  B8A	Red Edge 4			20 meters (good)
9  B9	Water vapor			60 meters (good)
10 B11	SWIR 1			    20 meters (moderate)
11 B12	SWIR 2			    20 meters (moderate)	
'''

# ANCHOR start major loop
filepath = 'input_images/data' # load

filepath_rgb = 'image_out/image' # save rgb
clip_percent = 5
# num_band = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) 
num_band = (4, 5, 6, 7, 8) # using only good bands
num_image = 11
# first row: band numbers, second row: number of image
store_water_pixel = np.zeros(shape=(num_image+1, 12), dtype=np.float32) 

for i_image in range (1, num_image+1):

	# TODO import data
	# Sentinel-2 MSI: MultiSpectral Instrument, Level-2A
	# print('image: ', i_image)
	stacked_images = skio.imread(filepath + (str(i_image).zfill(3)) + '.tif', plugin="tifffile")

	# TODO cleaning data contained NaN
	stacked_images, _, _ = c.cleaning_data(stacked_images)

	# TODO plot and save images
	# c.plot_imshow(stacked_images, 'test.svg')
	# c.plot_rgb(stacked_images, clip_percent, filepath_rgb, i_image)

	# TODO K-means (loop each band)
	# NOTE first inner loop
	for i in num_band:
		# print("compute kmeans of image: %s band: %s" % (i_image, i))
		kmeans = c.compute_kmeans(stacked_images, i, 3, 'no')

		# TODO remove unwanted areas using image segmentation
		print("compute IM of image: %s band: %s" % (i_image, i))
		if i == 4:
			min_pixel, max_pixel = 1, 500
			cleaned_kmeans = c.image_segmentation(kmeans, min_pixel, max_pixel, 'test.svg', i)
		else:
			min_pixel, max_pixel = 1, 500
			cleaned_kmeans = c.image_segmentation(kmeans, min_pixel, max_pixel, 'test.svg', i)

		# TODO count pixel
		pixel = c.count_number_of_pixel(cleaned_kmeans)
		print('water pixel of band', i, ':',  pixel)
		store_water_pixel[i_image, i] = pixel
print(store_water_pixel)
