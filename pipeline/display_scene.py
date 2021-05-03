import sys

volume_path = str(sys.argv[1])
segmentation_path = str(sys.argv[2])

print(str(sys.argv))
print(volume_path + 'eeeeeeeeee')
print(segmentation_path + 'eeeeeeeeee')

# Load volume
volume = slicer.util.loadVolume(volume_path)

# Load segmentation
segmentation = slicer.util.loadSegmentation(segmentation_path)

# Get first segment in segmentation (in our case, there should always only be one)
segment = segmentation.GetSegmentation().GetNthSegment(0)

# Extract name of segment 
segment_name = segment.GetName()

# Create segmentation display node
segmentationDisplayNode = segmentation.GetDisplayNode()

# Set visibility of segment (corresponding to tumor) to True (it is false by default)
segmentationDisplayNode.SetSegmentVisibility(segment_name, True)
