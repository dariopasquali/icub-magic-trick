from inaSpeechSegmenter import Segmenter, seg2csv
import time


media = './VAD/s3_1.mp3'
seg = Segmenter()


start = int(round(time.time() * 1000))
segmentation = seg(media)
delta = int(round(time.time() * 1000)) - start


print("Time %d" % delta)
seg2csv(segmentation, 'myseg.csv')