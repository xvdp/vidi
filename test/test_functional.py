
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from pytest import raises
from vidi.functional import _get_channels_axis, compress_fourcc, rgb2yxx, yxx_matrix
from vidi.functional import read_bytes, to_bits, from_bits, check_format

# pylint: disable=no-member
def test_get_channels_axis():
    # Test case for HWC shape
    shape = (480, 640, 3)
    num_channels = 3
    expected_output = 2
    assert _get_channels_axis(shape, num_channels)%len(shape) == expected_output

    # Test case for CHW shape
    shape = (3, 480, 640)
    num_channels = 3
    expected_output = 0
    assert _get_channels_axis(shape, num_channels)%len(shape) == expected_output

    # Test case for invalid shape
    shape = (480, 640, 4)
    num_channels = 3
    with raises(AssertionError):
        _get_channels_axis(shape, num_channels)


# def test_compress_fourcc():
#     # Test case for YUV 444
#     frame = np.random.randint(0, 255, size=(480, 640, 3))
#     fourcc = (b'Y', b'U', b'V', b' ')
#     expected_output = [
#         cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:, :, i] for i in range(3)
#     ]
#     assert_array_equal(compress_fourcc(frame, fourcc), expected_output)

#     # Test case for YUV 420
#     frame = np.random.randint(0, 255, size=(480, 640, 3))
#     fourcc = (b'Y', b'U', b'V', b'4')
#     expected_output = [
#         cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)[:, :, i], dsize=(320, 240))
#         for i in range(3)
#     ]
#     assert_array_equal(compress_fourcc(frame, fourcc), expected_output)

#     # Test case for invalid frame shape
#     frame = np.random.randint(0, 255, size=(480, 640, 4))
#     fourcc = (b'Y', b'U', b'V', b'4')
#     with raises(AssertionError):
#         compress_fourcc(frame, fourcc)




def test_yxx_matrix():
    # Test YUV_RGB_709 mode
    assert_allclose(yxx_matrix('YUV_RGB_709'),
                    np.array([[ 1.     ,  1.     ,  1.     ],
                              [ 0.     , -0.39465,  2.03211],
                              [ 1.13983, -0.5806 ,  0.     ]], dtype=np.float32))
    
    # Test YCC_RGB_709 mode
    assert_allclose(yxx_matrix('YCC_RGB_709'),
                    np.array([[ 1.    ,  1.    ,  1.    ],
                              [ 0.    , -0.1873,  1.8556],
                              [ 1.5748, -0.4681,  0.    ]], dtype=np.float32))
    
    # Test YCC_RGB_2020 mode
    assert_allclose(yxx_matrix('YCC_RGB_2020'),
                    np.array([[ 1.        ,  1.        ,  1.        ],
                              [ 0.        , -0.16455313,  1.8814    ],
                              [ 1.4746    , -0.57135313,  0.        ]], dtype=np.float32))
    
    # Test invalid mode
    try:
        yxx_matrix('invalid_mode')
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid mode did not raise AssertionError')

# def test_rgb2yxx():
#     # Test YUV_RGB_709 mode
#     rgb = np.array([[[0.5, 0.3, 0.2], [0.1, 0.7, 0.9]],
#                     [[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]]], dtype=np.float32)
#     yuv_expected = np.array([[[ 0.3278,  0.4366, -0.1645], [-0.1386,  0.5391,  0.4359]],
#                              [[ 0.3624,  0.4598, -0.0801], [ 0.7265, -0.1843, -0.5423]]], dtype=np.float32)
#     yuv_actual = rgb2yxx(rgb, mode='YUV_RGB_709')
#     assert_allclose(yuv_actual, yuv_expected)
    
#     # Test YCC_RGB_709 mode
#     rgb = np.array([[[0.5, 0.3, 0.2], [0.1, 0.7, 0.9]],
#                     [[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]]], dtype=np.float32)
#     ycc_expected = np.array([[[ 0.3278,  0.4366, -0.1645], [-0.1562,  0.4006,  0.3521]],
#                              [[ 0.3624,  0.4598, -0.0801], [ 0.6776, -0.0589, -0.3452]]], dtype=np.float32)

#     yuv_actual = rgb2yxx(rgb, mode='YCC_RGB_709')
#     assert_allclose(yuv_actual, ycc_expected)


# Tests for from_bits()
def test_from_bits():

    expected_output = np.array([[0., 0.4995112414467253], [1., 0.5004887585532747]])
    input_arr = np.array([[0, 511], [1023, 512]])
    output = from_bits(input_arr, 10)
    assert_array_almost_equal(output, expected_output)

# Tests for to_bits()
def test_to_bits():
    input_arr = np.array([[0., 0.4995112414467253], [1., 0.5004887585532747]])
    expected_output = np.array([[0, 511], [1023, 512]], dtype=np.uint16)
    output = to_bits(input_arr, 10)
    assert_array_almost_equal(output, expected_output)

# Tests for read_bytes()
def test_read_bytes():
    # Test with string input
    with TemporaryFile() as f:
        f.write(b'\x00\x01\x02\x03')
        f.seek(0)
        expected_output = b'\x00\x01\x02\x03'
        output = read_bytes(f.read())
        assert output == expected_output

    # Test with bytes input
    input_bytes = b'\x00\x01\x02\x03'
    output = read_bytes(input_bytes)
    assert output == input_bytes