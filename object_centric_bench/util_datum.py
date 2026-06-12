from io import BytesIO
import colorsys

import av
import numpy as np
import torch as pt
import torchvision.utils as ptvu


def rgb_segment_to_index_segment(segment_rgb: np.ndarray):
    """
    segment_rgb: shape=(h,w,c=3). r-g-b not b-g-r
    segment_idx: shape=(h,w)
    """
    assert segment_rgb.ndim == 3 and segment_rgb.dtype == np.uint8
    assert segment_rgb.shape[2] == 3
    segment0 = (segment_rgb * [[[256**0, 256**1, 256**2]]]).sum(2)
    segment_idx = (  # exactly same as the old implementation for-loop-assign
        np.unique(segment0, return_inverse=True)[1]
        .reshape(segment0.shape)
        .astype("uint8")
    )
    return segment_idx


def mask_segment_to_bbox_np(segment):
    """
    - segment: mask format, shape=(h,w,s)
    - bbox: ltrb format, shape=(s,c=4)
    """
    assert segment.ndim == 3 and segment.dtype == np.bool
    h, w, s = segment.shape
    y = np.arange(h)[:, None, None]
    x = np.arange(w)[None, :, None]
    l = np.amin(np.where(segment, x, np.inf), (0, 1))
    t = np.amin(np.where(segment, y, np.inf), (0, 1))
    r = np.amax(np.where(segment, x, -np.inf), (0, 1))
    b = np.amax(np.where(segment, y, -np.inf), (0, 1))
    bbox = np.stack([l, t, r, b], 1)
    valid = segment.any((0, 1))
    bbox[~valid] = 0
    bbox = bbox.astype("int32")
    # assert ((l <= r) & (t <= b)).all()  # has strange error for float64
    assert (bbox[:, :2] <= bbox[:, 2:]).all()  # left-closed and right-closed
    return bbox


def generate_spectrum_colors(num_color):
    spectrum = []
    for i in range(num_color):
        hue = i / float(num_color)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        spectrum.append([int(255 * c) for c in rgb])
    return np.array(spectrum, dtype="uint8")  # (s,c=3)


def draw_segmentation_np(image: np.ndarray, segment: np.ndarray, alpha=0.5, color=None):
    """
    - image: shape=(h,w,c)
    - segment: shape=(h,w,s), dtype=bool; in mask format, not index format
    - color: shape=(s,c=3)
    """
    h, w, c = image.shape
    h2, w2, s = segment.shape
    assert h == h2 and w == w2

    if color is None:
        color = generate_spectrum_colors(s)
    image2 = ptvu.draw_segmentation_masks(
        image=pt.from_numpy(image).permute(2, 0, 1),
        masks=pt.from_numpy(segment).permute(2, 0, 1),
        alpha=alpha,
        colors=color.tolist(),
    )
    return image2.permute(1, 2, 0).numpy()



class VideoCodec:
    """
    lossless encoding and decoding for videos

    - 3uint8: video (t,h,w,c=3) dtype=uint8 => no wrap, rgb24 => libx264rgb
    - 1uint16: depth (t,h,w,c=1) dtype=uint16 => wrap as rgb24 => libx264rgb

    Example
    ---
    ```
    video0 = np.random.randint(0, 255, [24, 256, 256, 3]).astype("uint8")
    # video0 = video0[:, :, :, :1].astype("uint16")  # 1uint16
    video0 = video0[:, :, :, :2].astype("uint16")  # 2uint16

    t20 = time()
    # buffer = VideoCodec.encode_3uint8(video0, fps)
    # buffer = VideoCodec.encode_1uint16(video0, fps)
    buffer = VideoCodec.encode_xuint16(video0, fps)
    # print(time() - t20, video0.nbytes, len(buffer.getvalue()))
    print(time() - t20, video0.nbytes, sum(len(_.getvalue()) for _ in buffer))

    t20 = time()
    # video2 = VideoCodec.decode_3uint8(buffer)
    # video2 = VideoCodec.decode_1uint16(buffer)
    video2 = VideoCodec.decode_xuint16(buffer)
    print(time() - t20)
    assert (video0 == video2).all()
    ```
    """

    @staticmethod
    def wrap_1uint16_as_rgb24(zero: np.ndarray) -> np.ndarray:
        t, h, w, c = zero.shape
        assert c == 1 and zero.dtype == np.uint16
        cr = (zero[:, :, :, 0] >> 8).astype("uint8")  # (t,h,w)
        cg = zero[:, :, :, 0].astype("uint8")
        cb = np.zeros_like(cg)
        wrap = np.stack([cr, cg, cb], axis=-1)  # (t,h,w,c=3)
        return wrap

    @staticmethod
    def wrap_rgb24_as_1uint16(zero: np.ndarray) -> np.ndarray:
        t, h, w, c = zero.shape
        assert c == 3 and zero.dtype == np.uint8
        c0 = zero[:, :, :, 0].astype("uint16") << 8 | zero[:, :, :, 1]  # (t,h,w)
        wrap = c0[:, :, :, None]  # (t,h,w,c=1)
        return wrap


    # 将单个样本的帧序列 (t,h,w,c) (numpy 数组)打包成一个内存中的完整单样本视频文件（avi）
    @staticmethod
    def encode_video_into_buffer(video: np.ndarray, pixfmt: str, fps: int) -> BytesIO:
        t, h, w, c = video.shape
        assert video.dtype == np.uint8
        if pixfmt == "rgb24":
            assert c == 3
            codec = "libx264rgb" # 无损转化的 RGB 格式
            stream_options = {
                "qp": "0",  # 压缩强度, 0 表示不压缩
                "tune": "fastdecode",  # 针对某类目标的优化策略，比如电影用film，这里的“快解码”更适合模型训练
                "preset": "ultrafast",  # 该格式下编码极快，文件稍大
            }
        elif pixfmt == "gray":
            assert c == 1
            codec = "ffv1"  # 一种数学无损视频编码方式
            stream_options = {
                "level": "3",  # 选择 level 表示需要更快的解码，而非更强的压缩
                "coder": "1",  # Range coder for exact lossless encoding
            }
        else:
            raise NotImplementedError

        buffer = BytesIO()  # 创建一个在内存中的二进制文件
        container = av.open(buffer, mode="w", format="avi") # 打开一个多媒体容器，可以看作“视频文件的外壳”，常见的有 MP4 格式，这里的 avi 更适合流式写入

        stream = container.add_stream(codec, rate=fps)  # 指定使用的编码格式 codec，以及视频的帧率 fps，返回一条已经配置好的编码通道，但此时还没喂任何帧
        stream.width = w   # 将视频长、宽等元信息存入 stream
        stream.height = h
        stream.pix_fmt = pixfmt
        stream.options = stream_options

        for frame_data in video:  # create a PyAV VideoFrame from the numpy array (RGB)
            frame = av.VideoFrame.from_ndarray(frame_data, format=pixfmt)  # 把 numpy 数组转成 PyAV 的 VideoFrame 对象

            for packet in stream.encode(frame):  # 把帧送进编码器
                container.mux(packet)  # 将编码好的 packet 写入容器

        # 把可能存在的全部缓冲帧全部输出
        for packet in stream.encode():  # flush the encoder
            container.mux(packet)

        container.close()
        buffer.seek(0)  # rewind to the start 让指针回到开头
        return buffer



    # 将一个内存中的完整单样本视频文件（avi）解码为单个样本的帧序列 (t,h,w,c) (numpy 数组)
    @staticmethod
    def decode_video_from_buffer(buffer: BytesIO) -> np.ndarray:
        container = av.open(buffer, mode="r", format="avi")

        frames = []
        # decode all frames from the first video stream
        for frame in container.decode(video=0):
            # convert frame to numpy array
            frame_array = frame.to_ndarray()  # format="rgb24"
            frames.append(frame_array)
        # list(map(...)) shows little speedup

        video = np.stack(frames, axis=0)  # (t,h,w,c)
        if video.ndim == 3:
            video = video[:, :, :, None]
        t, h, w, c = video.shape
        assert c in [1, 3] and video.dtype == np.uint8
        return video

    @staticmethod
    def encode_3uint8(video: np.ndarray, fps: int) -> BytesIO:
        return __class__.encode_video_into_buffer(video, "rgb24", fps)

    @staticmethod
    def decode_3uint8(buffer: BytesIO) -> np.ndarray:
        return __class__.decode_video_from_buffer(buffer)

    @staticmethod
    def encode_1uint8(video: np.ndarray, fps: int) -> BytesIO:
        return __class__.encode_video_into_buffer(video, "gray", fps)

    @staticmethod
    def decode_1uint8(buffer: BytesIO) -> np.ndarray:
        return __class__.decode_video_from_buffer(buffer)

    @staticmethod
    def encode_1uint16(zero: np.ndarray, fps: int) -> BytesIO:
        wrap = __class__.wrap_1uint16_as_rgb24(zero)  # assert inside
        buff = __class__.encode_3uint8(wrap, fps)
        return buff

    @staticmethod
    def decode_1uint16(buff: BytesIO) -> np.ndarray:
        wrap = __class__.decode_3uint8(buff)
        zero = __class__.wrap_rgb24_as_1uint16(wrap)  # assert inside
        return zero

    @staticmethod
    def encode_xuint16(zero: np.ndarray, fps: int) -> list:
        zero = np.split(zero, zero.shape[-1], axis=-1)
        buff = [__class__.encode_1uint16(_, fps) for _ in zero]
        return buff

    @staticmethod
    def decode_xuint16(buff: list) -> np.ndarray:
        zero = [__class__.decode_1uint16(_) for _ in buff]
        zero = np.concatenate(zero, axis=-1)  # (t,h,w,x)
        return zero
