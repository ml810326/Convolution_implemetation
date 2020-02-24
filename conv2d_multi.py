# Reference from 
import numpy as np

def conv2d_multi_channel(input, w):
    """Two-dimensional convolution with multiple channels.

    Uses SAME padding with 0s, a stride of 1 and no dilation.

    input: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth, out_depth) with odd fd.
       in_depth is the number of input channels, and has the be the same as
       input's in_depth; out_depth is the number of output channels.

    Returns a result with shape (height, width, out_depth).
    """
    assert w.shape[0] == w.shape[1] and w.shape[0] % 2 == 1

    padw = w.shape[0] // 2
    padded_input = np.pad(input,
                          pad_width=((padw, padw), (padw, padw), (0, 0)),
                          mode='constant',
                          constant_values=0)

    height, width, in_depth = input.shape
    assert in_depth == w.shape[2]
    out_depth = w.shape[3]
    output = np.zeros((height, width, out_depth))

    count = 0
    for out_c in range(out_depth):
        # For each output channel, perform 2d convolution summed across all
        # input channels.
        for i in range(height):
            for j in range(width):
                # Now the inner loop also works across all input channels.
                for c in range(in_depth):
                    for fi in range(w.shape[0]):
                        for fj in range(w.shape[1]):
                            w_element = w[fi, fj, c, out_c]
                            output[i, j, out_c] += (padded_input[i + fi, j + fj, c] * w_element)
                            count = count + 1
    print(count)
    return output

coordinates = np.random.randint(0, 255, size=(8, 8, 3))
mask = np.random.randint(0, 2, size=(3, 3, 3, 2))
test = conv2d_multi_channel(coordinates, mask)
#print(test)
