def lenArrayOfDataframe(ArrayOfDataframe):

	length = 0

	for i in range(len(ArrayOfDataframe)):

		length = length + len(ArrayOfDataframe[i])

	return length