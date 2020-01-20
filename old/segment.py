features = []
segmentations = []
labelsSegmentationRepetitions = []

# Loop over repetitions
for i in range(nRepetitions):

    if SETTINGS.VERBOSE_LEVEL >= 2:
        print('REPETITION ' + str(i) + '/' + str(nRepetitions))
    else:
        pass

    # (2) Segmentation

    segmentationClass = Segmentation()
    segmentationFunction = SETTINGS.SEGMENTATION_TECHNIQUE['method']
    print("Segmentation {}".format(segmentationFunction))
    print("\n")

    try:
        segmentationMethod = getattr(segmentationClass, segmentationFunction)
    except AttributeError:
        raise NotImplementedError(
            "Class `{}` does not implement `{}`".format(segmentationClass.__class__.__name__,
                                                        segmentationFunction))
    segmentDf = segmentationMethod(data[i].values, SETTINGS)
    segmentations.append(segmentDf)
    labelsSegmentations = assignLabels(segments[i], segmentations[i])
    labelsSegmentationRepetitions.append(labelsSegmentations)

    # (3) Feature extraction
    feat, fType, fDescr = featureExtraction(data[i], segmentations[i], SETTINGS.FEATURE_TYPE, SETTINGS.VERBOSE_LEVEL)
    features.append(feat)