from pathlib import Path
import json
import pickle as pkl
import time
import av
import io
import random

from einops import rearrange

import cv2
import lmdb
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util_datum import draw_segmentation_np, mask_segment_to_bbox_np

class CLEVRER_VQA(ptud.Dataset):

    def __init__(
            self,
            data_file,
            vocab_file,
            extra_keys=["question","segment","bbox"],
            transform=lambda **_: _,
            base_dir: Path = None,
            video_len = 128,
            max_question_len=20,  # for padding
            max_choice_len=12,  # for padding
            sample_frames=25,  # 稀疏采样视频后的帧数
            sample_offset=None,
            sample_mode="train",  # ["train","val"]
    ):

        if base_dir:
            data_file = base_dir / data_file
            vocab_file = base_dir / vocab_file
        self.data_file = data_file

        # 载入各种映射表
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        self.q_vocab = vocab['q_vocab']
        self.answer2label = vocab['a_vocab']
        self.label2answer = {v: k for k, v in self.answer2label.items()}
        self.q_subtype2id = {
            'descriptive': 0,
            'explanatory': 1,
            'predictive': 2,
            'counterfactual': 3,
        }
        self.q_type2id = {
            'descriptive': 0,  # cls
            'explanatory': 1,  # mc
            'predictive': 1,  # mc
            'counterfactual': 1,  # mc
        }

        # 随机起点采样 + 固定步长的视频稀疏采样
        self.video_len = video_len
        self.sample_frames = sample_frames
        self.sample_offset = video_len // sample_frames if sample_offset is None else sample_offset
        max_start_idx = video_len - (sample_frames - 1) * self.sample_offset
        self.start_idx = list(range(max_start_idx))

        self.max_question_len = max_question_len
        self.max_choice_len = max_choice_len

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self_keys = pkl.loads(txn.get(b"__keys__"))
        print(len(self_keys))
        self.keys = []
        print(f"[{__class__.__name__}] map samples in dataset...")
        t0 = time.time()
        for key in self_keys:
            with env.begin(write=False) as txn:
                sample = pkl.loads(txn.get(key))
                q_num = len(sample["question"])
                for q_id in range(q_num):
                    self.keys.append([key, q_id])
        print(f"[{__class__.__name__}] {time.time() - t0}")
        env.close()

        self.extra_keys = extra_keys
        self.transform = transform
        assert sample_mode in ["train","val"]
        self.sample_mode = sample_mode


    def __getitem__(self, index):
        """
        video: Tensor (t,c,h,w) uint8 | float32
        segment: Tensor (t,h,w,s) bool
        bbox: Tensor (t,s,c=4) float32

        question:{
                - video_index: Tensor, () —— 标量 Tensor, x.ndim = 0
                - question_id: Tensor, ()
                - q_type: Tensor, (), 0 as cls q while 1 as mc q
                - q_subtype: Tensor, (), 0, 1, 2, 3 as 4 subtypes of questions
                - raw_question: str
                - q_tokens:
                    in cls question, Tensor, (L q+c,)
                    in mc question, Tensor, (num_choices, Lq + Lc)
                - q_pad_mask:
                    in cls question, BoolTensor, (L q+c,)
                    in mc question, BoolTensor, (num_choices, Lq + Lc)
                - mc_raw_choices(only in mc question): list of str
                - mc_choice_id (only in mc question): Tensor, (num_choices,)
                - mc_flag (only in mc question):an all_zeros Tensor of shape (num_choices, )
                - answer:
                    in cls question,  Tensor, ();
                    in mc question, Tensor, (num_choices,)
        }
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        if self.sample_mode == "train":
            st = np.random.choice(self.start_idx)
        else:
            st = self.start_idx[len(self.start_idx) // 2]

        frame_indices = [st + n * self.sample_offset for n in range(self.sample_frames)]

        key, q_id = self.keys[index]
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(key))
        sample1 = {}

        try:
            video = sample0["video"]
            if len(video) != self.video_len:
                raise ValueError

            frames = []
            for i in frame_indices:
                frame = video[i]
                if frame is None:
                    raise ValueError
                frame = cv2.imdecode(
                    np.frombuffer(frame, np.uint8),
                    cv2.IMREAD_COLOR
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        except Exception as e:
            raise RuntimeError(
                f"[CLEVRER_VQA ERROR]\n"
                f"index={index}\n"
                f"key={key.decode()}\n"
                f"data_file={self.data_file}\n"
                f"original_error={repr(e)}"
            ) from e
        # except ValueError:
        #     return self.__getitem__(np.random.randint(0, len(self.keys)))

        video = np.stack(frames, axis=0)
        video = rearrange(pt.from_numpy(video), 't h w c -> t c h w')
        sample1["video"] = video  # (t,c,h,w) uint8


        if "segment" in self.extra_keys:
            segment = sample0["segment"]
            segment = np.array([cv2.imdecode(_, cv2.IMREAD_GRAYSCALE) for _ in segment])
            segment = pt.from_numpy(segment)
            segment = segment[frame_indices]
            sample1["segment"] = segment  # (t,h,w) uint8

        sample2 = self.transform(**sample1)

        if "question" in self.extra_keys:
            qi= sample0["question"][q_id]
            q_type = self.q_type2id[qi['question_type']]
            raw_question = qi['question']  # str
            if q_type == 0:  # cls
                q_tokens, q_pad_mask = self._tokenize_text(raw_question, self.q_vocab, self.max_question_len + self.max_choice_len)
                if 'answer' in qi:
                    cls_answer = int(self.answer2label[qi['answer']])
                else:
                    cls_answer = -1  # test set
            else:  # mc
                q_tokens, q_pad_mask = self._tokenize_text(raw_question, self.q_vocab, self.max_question_len)
                c_tokens = []
                c_pad_mask = []
                mc_answer = []
                for choice in qi['choices']:
                    tok, mask = self._tokenize_text(choice['choice'], self.q_vocab, self.max_choice_len)
                    c_tokens.append(tok)
                    c_pad_mask.append(mask)
                    if 'answer' in choice:
                        mc_answer.append(choice['answer'] == 'correct')
                    else:
                        mc_answer.append(-1)  # test set
            qi_dict = {
                'video_index': pt.tensor(int(key), dtype=pt.int64),
                'question_id': pt.tensor(q_id, dtype=pt.int64),
                'q_type': pt.tensor(q_type, dtype=pt.int64),
                'q_subtype': pt.tensor(self.q_subtype2id[qi['question_type']], dtype=pt.int64),
                'raw_question': raw_question,
                'q_tokens':pt.stack([pt.cat([q_tokens, ctk], dim=0)for ctk in c_tokens]) if q_type==1 else q_tokens,
                'q_pad_mask':pt.stack([pt.cat([q_pad_mask, cpm], dim=0)for cpm in c_pad_mask]) if q_type==1 else q_pad_mask,

                'mc_choice_id':pt.tensor([choice['choice_id'] for choice in qi['choices']], dtype=pt.int64) if q_type==1 else None,
                'mc_raw_choices':[choice['choice'] for choice in qi['choices']] if q_type==1 else None,
                'mc_flag':pt.zeros_like(pt.tensor(mc_answer,dtype=pt.int64)) if q_type==1 else None,

                'answer': pt.tensor(mc_answer,dtype=pt.int64) if q_type==1 else pt.tensor(cls_answer, dtype=pt.int64),
            }

            sample2["question"] = qi_dict

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

        sample2["frame_indices"] = frame_indices  # just for transmitting to CLEVRER_VQA_Slots
        return sample2


    def __len__(self):
        return len(self.keys)


    def analyze_question_choice_lengths(self):
        """
        Analyze token length statistics for questions and choices
        to help determine reasonable max_question_len and max_choice_len.
        """
        env = lmdb_open_read(self.data_file)

        q_lens = []  # question token lengths
        c_lens = []  # choice token lengths (MC only)
        for key, q_id in self.keys:
            with env.begin(write=False) as txn:
                sample = pkl.loads(txn.get(key))

            qi = sample["q_list"][q_id]
            q_len = len([word for word in qi["question"].lower().replace('?', '').split(' ') if word])
            q_lens.append(q_len)

            if qi["question_type"] != "descriptive":  # MC question
                for choice in qi["choices"]:
                    c_len = len([word for word in choice["choice"].lower().replace('?', '').split(' ') if word])
                    c_lens.append(c_len)

        env.close()

        def stat(arr):
            arr = np.array(arr)
            return {
                "max": int(arr.max()),
                "mean": float(arr.mean()),
                "p95": int(np.percentile(arr, 95)),
                "p99": int(np.percentile(arr, 99)),
            }

        stats = {
            "question_len": stat(q_lens),
            "choice_len": stat(c_lens) if c_lens else None,
        }

        for k, v in stats.items():
            print(f"{k}: {v}")

        return stats


    @staticmethod
    def _tokenize_text(text, vocab, pad_to_length):
        """Convert a question str to a 1d np array and do the padding."""

        text = text.lower().replace('?', '').split(' ')
        tokens = [vocab[word] for word in text if word]  # eliminate ''

        assert pad_to_length >= len(tokens)
        tokens_pt = pt.full(
            (pad_to_length,),
            fill_value=vocab['PAD'],
            dtype=pt.int32
        )
        tokens_pt[:len(tokens)] = pt.tensor(tokens, dtype=pt.int32)
        pad_mask = pt.ones(pad_to_length, dtype=pt.bool)
        pad_mask[:len(tokens)] = False
        return tokens_pt, pad_mask


    @staticmethod
    def convert_dataset(
            src_dir=Path("../datasets/CLEVRER_origin/CLEVRER"),
            dst_dir=Path("../datasets/clevrer"),
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


class ClevrerCollate:
    def __call__(self, list_data: list) -> dict:
        """
        input:
                video: Tensor (t,c,h,w) uint8 | float32
                segment: Tensor (t,h,w,s) bool
                bbox: Tensor (t,s,c=4) float32
                pre_computed:{
                    - sloz: Tensor (t,n,c) float32
                    - attent2: Tensor (t,n,h,w) float32
                }

                question:{
                    - video_index: Tensor, () —— 标量 Tensor, x.ndim = 0
                    - question_id: Tensor, ()
                    - q_type: Tensor, (), 0 as cls q while 1 as mc q
                    - q_subtype: Tensor, (), 0, 1, 2, 3 as 4 subtypes of questions
                    - raw_question: str
                    - q_tokens:
                        in cls question, Tensor, (L q+c,)
                        in mc question, Tensor, (num_choices, Lq + Lc)
                    - q_pad_mask:
                        in cls question, BoolTensor, (L q+c,)
                        in mc question, BoolTensor, (num_choices, Lq + Lc)
                    - mc_raw_choices(only in mc question): list of str
                    - mc_choice_id (only in mc question): Tensor, (num_choices,)
                    - mc_flag (only in mc question):an all_zeros Tensor of shape (num_choices, )
                    - answer:
                        in cls question,  Tensor, ();
                        in mc question, Tensor, (num_choices,)
                }

        output:
                video: Tensor (b,t,c,h,w) uint8
                segment: Tensor (b,t,h,w,s) bool
                bbox: Tensor (b,t,s,c=4) float32
                pre_computed:{
                    - sloz: Tensor (b,t,n,c) float32
                    - attent2: Tensor (b,t,n,h,w) float32
                }

                question:{
                    - video_index: Tensor, (b,)
                    - question_id: Tensor, (b,)
                    - question_type: Tensor, (b,)

                    - cls_q_tokens: Tensor, (b1, L,)
                    - cls_q_pad_mask: Tensor, (b1, L,)
                    - cls_label: Tensor, (b1,)

                    - mc_subtype: Tensor, (b2, )
                    - mc_q_tokens: Tensor, (b2n, L), concated along num_choices dim
                    - mc_q_pad_mask: Tensor, (b2n, L)
                    - mc_label: Tensor, (b2n,)
                    - mc_flag: Tensor, (b2n,) e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
                    - mc_choice_id: Tensor, (b2n,) e.g. [0, 1, 2, 0, 1, 2, 3, 0, 1, ...]
                }
        """

        def pt_stack(arrays, dim=0):
            if not arrays:
                return pt.tensor([])
            return pt.stack(arrays, dim=dim)

        def pt_cat(arrays, dim=0):
            if not arrays:
                return pt.tensor([])
            return pt.cat(arrays, dim=dim)

        batch_data = {}
        batch_data['video'] = pt_stack([data['video'] for data in list_data])

        if 'segment' in list_data[0].keys():
            batch_data['segment'] = pt_stack([data['segment'] for data in list_data])

        if 'bbox' in list_data[0].keys():
            batch_data['bbox'] = pt_stack([data['bbox'] for data in list_data])

        if 'pre_computed' in list_data[0].keys():
            batch_data['pre_computed'] = {}
            pre_computed_keys = list_data[0]['pre_computed'].keys()
            for k in pre_computed_keys:
                batch_data['pre_computed'][k] = pt_stack(
                    [data['pre_computed'][k] for data in list_data]
                )

        if 'question' in list_data[0].keys():
            cls_data = [data for data in list_data if data["question"]['q_type'] == 0]
            mc_data = [data for data in list_data if data["question"]['q_type'] == 1]
            # pack cls and mc questions separately
            num_mc = len(mc_data)

            batch_data['question'] = {
                'video_index': pt_stack([data['question']['video_index'] for data in list_data]),
                'question_id': pt_stack([data['question']['question_id'] for data in list_data]),
                'question_type': pt_stack([data['question']['q_type'] for data in list_data]),

                'cls_q_tokens': pt_stack([data['question']['q_tokens'] for data in cls_data]),
                'cls_q_pad_mask': pt_stack([data['question']['q_pad_mask'] for data in cls_data]),
                'cls_label': pt_stack([data['question']['answer'] for data in cls_data]),

                'mc_subtype': pt_stack([data['question']['q_subtype'] for data in mc_data]),
                'mc_q_tokens': pt_cat([data['question']['q_tokens'] for data in mc_data]),
                'mc_q_pad_mask': pt_cat([data['question']['q_pad_mask'] for data in mc_data]),
                'mc_label': pt_cat([data['question']['answer'] for data in mc_data]),
                'mc_flag': pt_cat([mc_data[i]['question']['mc_flag'] + i for i in range(num_mc)]),
                'mc_choice_id': pt_cat([data['question']['mc_choice_id'] for data in mc_data]),
            }

        return batch_data


class CLEVRER_VQA_Slots(CLEVRER_VQA):
    """ Dataset for loading CLEVRER VQA video embs and QA pairs. """
    def __init__(
        self,
        data_file,
        vocab_file,
        slotz_file,
        extra_keys=["question", "segment", "bbox"],
        pre_computed_keys=["slotz"],  # ["feature", "attent2", "recon"]
        transform=lambda **_: _,
        base_dir: Path = None,
        video_len=128,
        max_question_len=20,  # for padding
        max_choice_len=12,  # for padding
        sample_frames=25,  # 稀疏采样视频后的帧数
        sample_offset=None,
        sample_mode="train",  # ["train","val"]
    ):
        super().__init__(
            data_file=data_file,
            vocab_file=vocab_file,
            extra_keys=extra_keys,
            transform=transform,
            base_dir=base_dir,
            video_len=video_len,
            max_question_len=max_question_len,
            max_choice_len=max_choice_len,
            sample_frames=sample_frames,
            sample_offset=sample_offset,
            sample_mode=sample_mode,
        )

        if base_dir:
            slotz_file = base_dir / slotz_file
        self.slotz_file = slotz_file
        assert "slotz" in pre_computed_keys
        self.pre_computed_keys = pre_computed_keys


    def __getitem__(self, index):
        """
        Data dict (added compared to its super class):
            pre_computed:{
                - sloz: [t, n, c]
                - attent2: [t, n, h, w]  可选
            }
        """
        if not hasattr(self, "env_slotz"):  # torch>2.6
            self.env_slotz = lmdb_open_read(self.slotz_file)

        data_dict = super().__getitem__(index)

        try:
            key, _ = self.keys[index]
            with self.env_slotz.begin(write=False) as txn:
                sample = pkl.loads(txn.get(key))
            frame_indices = data_dict["frame_indices"]

            pre_computed = {}
            for k in self.pre_computed_keys:
                pre_computed[k] = pt.from_numpy(sample[k][frame_indices])
            data_dict["pre_computed"] = pre_computed

        except Exception as e:
            raise RuntimeError(
                f"[CLEVRER_VQA_Slots ERROR]\n"
                f"index={index}\n"
                f"key={key.decode()}\n"
                f"slotz_file={self.slotz_file}\n"
                f"original_error={repr(e)}"
            ) from e

        # except Exception:
        #     return self.__getitem__(np.random.randint(0, len(self.keys)))

        return data_dict

