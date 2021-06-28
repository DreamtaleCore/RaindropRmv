from os.path import basename
import glob
import numpy as np
import cv2

key = lambda x: int(basename(x).split('_')[0])

#mask = sorted(glob.glob('../result/ardcnn_all/*'))
#rain = sorted(glob.glob('/home/zx/repo/dataset/rain_test/youtube/geko_all/*_B.png'))
#icnn = sorted(glob.glob('../result/icnn_all/*'))

mask = sorted(glob.glob('/home/zx/repo/dataset/rain_result/ours/allover/mask/*'))
rain = sorted(glob.glob('/home/zx/repo/dataset/rain_test/allover/*_B.png'), key=key)
icnn = sorted(glob.glob('/home/zx/repo/dataset/rain_result/ours/allover/icnn/*'))

savePath = '/home/zx/repo/dataset/rain_result/ours/allover/combine/'

assert len(mask) == len(rain) == len(icnn)

dilation_size = 3
element = cv2.getStructuringElement(2, (dilation_size, dilation_size))

for index, (m, r, i) in enumerate(zip(mask, rain, icnn)):
    img_m = cv2.imread(m)
    #img_m = dilate(3, img_m)
    #img_m = cv2.dilate(img_m, element)
    img_r = cv2.imread(r)
    img_i = cv2.imread(i)

    img = np.where(img_m == 255, img_i, img_r)
    cv2.imwrite((savePath + '%04d.png') % index, img)
    #cv2.imwrite((savePath + '%04d_m.png') % index, img_m)

print('Done!!!')
