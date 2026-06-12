from pathlib import Path
import json
import pickle as pkl
import time

from einops import rearrange

import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util_datum import draw_segmentation_np, mask_segment_to_bbox_np


class CLEVRER_Video(ptud.Dataset):

    def __init__(
            self,
            data_file,
            extra_keys=["segment", "bbox"],
            transform=lambda **_: _,
            base_dir: Path = None,
            video_len=128,
            sample_frames=6,  # 稀疏采样视频后的帧数
            sample_offset=None,
            sample_mode="train",  # ["train","val","extract"]
    ):

        if base_dir:
            data_file = base_dir / data_file
        self.data_file = data_file

        self.video_len = video_len
        self.sample_frames = sample_frames
        self.sample_offset = video_len // sample_frames if sample_offset is None else sample_offset

        assert sample_mode in ["train", "val", "extract"]
        if sample_mode == "train":
            max_start = video_len - (sample_frames - 1) * self.sample_offset
            start_idx = list(range(max_start))
            frame_idx = [
                [start + i * self.sample_offset for i in range(sample_frames)]
                for start in start_idx
            ]
        elif sample_mode == "val":
            size = sample_frames * self.sample_offset
            start_idx = []
            for i in range(0, self.video_len - size + 1, size):
                for j in range(self.sample_offset):
                    start_idx.append(i + j)
            frame_idx = [
                [start + i * self.sample_offset for i in range(sample_frames)]
                for start in start_idx
            ]
        else:
            frame_idx = [list(range(0, self.video_len, self.sample_offset))]

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self_keys = pkl.loads(txn.get(b"__keys__"))
        self.keys = []
        for key in self_keys:
            for f_idx in frame_idx:
                self.keys.append([key, f_idx])
        env.close()

        self.extra_keys = extra_keys
        self.transform = transform

    def __getitem__(self, index):
        """
        video: (t,c,h,w) uint8 | float32
        segment: (t,h,w,s) bool
        bbox: (t,s,c=4) float32
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        key, f_idx = self.keys[index]
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(key))
        sample1 = {}

        try:
            video = sample0["video"]
            if len(video) != self.video_len:
                raise ValueError

            frames = []
            for i in f_idx:
                frame = video[i]
                if frame is None:
                    raise ValueError
                frame = cv2.imdecode(
                    np.frombuffer(frame, np.uint8),
                    cv2.IMREAD_COLOR
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        except ValueError:
            return self.__getitem__(np.random.randint(0, len(self.keys)))

        video = np.stack(frames, axis=0)
        video = rearrange(pt.from_numpy(video), 't h w c -> t c h w')
        sample1["video"] = video  # (t,c,h,w) uint8

        if "segment" in self.extra_keys:
            segment = sample0["segment"]
            segment = np.array([cv2.imdecode(_, cv2.IMREAD_GRAYSCALE) for _ in segment])
            segment = pt.from_numpy(segment)
            segment = segment[f_idx]
            sample1["segment"] = segment  # (t,h,w) uint8

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # (h,w) index format
            segment3 = ptnf.one_hot(segment2.long()).bool()

            t, h, w, s = segment3.shape

            # ``RandomCrop`` and ``CenterCrop`` can diminish segments
            cond = segment3.any([0, 1, 2])  # (s,)

            segment3 = segment3[:, :, :, cond]
            sample2["segment"] = segment3  # (t,h,w,s) bool

            if "bbox" in self.extra_keys:
                segment3_ = rearrange(
                    segment3[:, :, :, 1:], "t h w s -> h w (t s)"
                )  # [:, :, :, 1:] skip bg
                bbox2_ = pt.from_numpy(  # (t*s,c=4)
                    mask_segment_to_bbox_np(segment3_.numpy())
                ).float()
                bbox2 = rearrange(bbox2_, "(t s) c -> t s c", t=t)
                bbox2[:, :, 0::2] /= w  # normalize
                bbox2[:, :, 1::2] /= h
                sample2["bbox"] = bbox2  # (t,s,c=4) float32

        return sample2

    def __len__(self):
        return len(self.keys)


    # 占用空间 train：32 GB、 val/test:16 GB  todo jpg to png
    # 执行 savi_aloe-clevrer_video-mean_std.py 单周期大约用时 4.5 h
    # 先将 MP4 解码为 jpg 序列, 再存入 lmdb, 该方法所占的存储空间适中, 运行速度最快 (解决了 num workers 的卡顿), 唯一的缺点是解码原始的 MP4 得到的 jpg 也会占用存储空间
    @staticmethod
    def convert_dataset(
            src_dir=Path("./datasets/clevrer_raw"),
            dst_dir=Path("./datasets/clevrer"),
    ):
        from pycocotools import mask as mask_utils

        dst_dir.mkdir(parents=True, exist_ok=True)

        splits = dict(
            train=[0, 10000],
            val=[10000, 15000],
            test=[15000, 20000],
        )
        group_num = 1000

        video_fold = src_dir / "videos"
        q_fold = src_dir / "questions"
        segment_fold = src_dir / "derender_proposals"

        for split, [start, end] in splits.items():
            q_file = q_fold / f"{split}.json"
            with open(q_file, "r") as f:
                q_json = json.load(f)

            lmdb_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb_open_write(lmdb_file)

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            cnt = 0
            for group_start in range(start, end, group_num):
                group_end = group_start + group_num
                print(split, group_start, group_end)

                for idx in range(group_start, group_end):
                    try:
                        video_file = video_fold / f"{split}" / f"video_{group_start:05d}-{group_end:05d}" / f"video_{idx:05d}.mp4"
                        frame_dir = video_file.with_suffix("")  # 去掉'.mp4'后缀
                        frame_dir.mkdir(parents=True, exist_ok=True)

                        # 将视频解码为 jpg 序列
                        if len(list(frame_dir.glob("*.jpg"))) == 128:
                            pass
                        else:
                            cap = cv2.VideoCapture(str(video_file))
                            frame_idx = 0
                            while True:
                                ret, frame = cap.read()  # 当 ret 为 False → 没有读到帧 → 视频结束
                                if not ret:
                                    break

                                frame_path = frame_dir / f"{frame_idx:04d}.jpg"
                                if frame_path.exists():
                                    frame_idx += 1
                                    continue

                                cv2.imwrite(
                                    str(frame_path),
                                    frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                                )
                                frame_idx += 1
                            cap.release()

                        if frame_idx != 128:
                            raise ValueError(f"invalid video frame num: {frame_idx}")

                        video_b = [
                            (frame_dir / f"{i:04d}.jpg").read_bytes()
                            for i in range(128)
                        ]

                        question = q_json[idx - start]['questions']

                        segment_file = segment_fold / f"proposal_{idx:05d}.json"
                        with open(segment_file, "r") as f:
                            segment_json = json.load(f)
                        frames_info = segment_json["frames"]

                        t = len(frames_info)
                        h, w = frames_info[0]["objects"][0]["mask"]["size"]
                        if not (t == 128 and h == 320 and w == 480):
                            raise ValueError(f"invalid segment shape: t={t}, h={h}, w={w}")

                        segment = np.zeros([t, h, w], "uint8")
                        for f_id, f_info in enumerate(frames_info):
                            objects = f_info["objects"]
                            for obj_id, obj in enumerate(objects, start=1):  # 不包含背景
                                mask_rle = {
                                    "size": obj["mask"]["size"],
                                    "counts": obj["mask"]["counts"]
                                }
                                mask = mask_utils.decode(mask_rle)  # (h, w), bool
                                segment[f_id][mask == 1] = obj_id  # (t, h, w) uint8
                    except ValueError as e:
                        print(f"[skip] idx={idx}: {e}")
                        continue

                    sample_key = f"{cnt:06d}".encode("ascii")
                    keys.append(sample_key)
                    sample_dict = dict(
                        video=video_b,  # (t,h,w,c=3) bytes
                        question=question,  # python list
                        segment=[  # (t,h,w) bytes
                            cv2.imencode(".webp", _)[1] for _ in segment
                        ],
                    )
                    txn.put(sample_key, pkl.dumps(sample_dict))

                    if (cnt + 1) % 64 == 0:  # write_freq
                        print(f"{cnt + 1:06d}")
                        txn.commit()
                        txn = lmdb_env.begin(write=True)

                    cnt += 1

            txn.commit()
            print((time.time() - t0) / cnt)

            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()


    @staticmethod
    def visualiz(video, segment=None, bbox=None, wait=0):
        """
        - video: bgr format, shape=(t,h,w,c=3), uint8
        - segment: index format, shape=(t,h,w,s), bool
        - bbox: both side normalized ltrb, shape=(t,s,c=4), float32
        """
        assert video.ndim == 4 and video.shape[3] == 3 and video.dtype == np.uint8

        if segment is not None:
            assert segment.ndim == 4 and segment.dtype == bool

        if bbox is not None and bbox.shape[0]:
            assert bbox.ndim == 3 and bbox.shape[2] == 4 and bbox.dtype == np.float32
            t, h, w, c = video.shape
            bbox[:, :, 0::2] *= w
            bbox[:, :, 1::2] *= h
            bbox = np.round(bbox).astype("int")

        c1 = (63, 127, 255)
        imgs = []
        segs = []

        for t, img in enumerate(video):
            if bbox is not None and len(bbox) > 0:
                for b in bbox[t]:
                    cv2.rectangle(img, b[:2], b[2:], color=c1)

            cv2.imshow("v", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            imgs.append(img)

            if segment is not None:
                seg = draw_segmentation_np(img, segment[t], alpha=1)
                cv2.imshow("s", cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
                segs.append(seg)

            cv2.waitKey(wait)

        return imgs, segs




