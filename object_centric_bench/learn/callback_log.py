from pathlib import Path
import json

import numpy as np
import torch as pt
import pickle as pkl

from ..util import DictTool
from .callback import Callback

from ..datum import lmdb_open_write



class AverageLog(Callback):
    """"""

    def __init__(
        self, log_file=None, epoch_key="epoch", loss_key="loss", acc_key="acc"
    ):
        super().__init__()
        self.log_file = log_file
        self.epoch_key = epoch_key
        self.loss_key = loss_key
        self.acc_key = acc_key
        self.idx = None
        self.state_dict = {}

    @pt.inference_mode()
    def index(self, epoch, isval):
        self.idx = f"{epoch}" if not isval else f"{epoch}/val"
        self.state_dict.clear()

    @pt.inference_mode()
    def append(self, loss, acc):
        for k, v in {**loss, **acc}.items():
            assert len(v) == 2  # (loss/acc,valid)
            v = tuple([_.detach().cpu().numpy() for _ in v])
            if k in self.state_dict:
                self.state_dict[k].append(v)
            else:
                self.state_dict[k] = [v]

    @pt.inference_mode()
    def mean(self):
        avg_dict = {}
        for k, v in self.state_dict.items():
            metric, valid = list(zip(*v))
            metric = np.concatenate(metric, 0)  # concat all batches  # (b,..)
            valid = np.concatenate(valid, 0)  # (b,)
            metric2 = metric[valid]  # keep the valid ones  # (?,..)
            v2 = np.mean(metric2, 0)  # .round(8) can cause nan
            avg_dict[k] = v2.tolist()
        if self.log_file:
            __class__.save(self.idx, avg_dict, self.log_file)
        print(self.idx, avg_dict)
        return avg_dict

    @pt.inference_mode()
    @staticmethod
    def save(key, avg_dict, log_file):
        line = json.dumps({key: avg_dict})
        with open(log_file, "a") as f:
            f.write(line + "\n")

    @pt.inference_mode()
    def before_epoch(self, isval, **pack):
        epoch = pack[self.epoch_key]
        self.index(epoch, isval)
        return pack

    @pt.inference_mode()
    def after_step(self, **pack):
        loss = pack[self.loss_key]
        acc = pack[self.acc_key]
        self.append(loss, acc)
        return pack

    @pt.inference_mode()
    def after_epoch(self, **pack):
        self.mean()
        return pack


class HandleLog(Callback):
    """Generalized logger:
    - can record anything into log;
    - can do any operation on the recorded logs.
    """

    def __init__(
        self, log_file=None, epoch_key="epoch", ikeys=[[]], okeys=None, ops=[""]
    ):
        super().__init__()

        self.log_file = log_file
        self.epoch_key = epoch_key

        assert (
            isinstance(ikeys, list)
            and all(isinstance(_, list) for _ in ikeys)
            and all(all(isinstance(__, str) for __ in _) for _ in ikeys)
        )
        assert len(ikeys) == len(ops)

        self.ikeys = ikeys  # list of list of str
        if okeys is None:
            okeys = ikeys
        self.okeys = okeys  # list of list of str

        assert isinstance(ops, list) and all(isinstance(_, str) for _ in ops)
        self.ops = ops  # list of str

        self.state_dict = {}

    @pt.inference_mode()
    def index(self, epoch, isval):
        self.idx = f"{epoch}" if not isval else f"{epoch}/val"
        self.state_dict.clear()

    @pt.inference_mode()
    def append(self, kwds: dict):
        for k, v in kwds.items():
            assert len(v) == 2  # [loss/acc,valid]
            v = tuple(
                _.detach().cpu().numpy() if isinstance(_, pt.Tensor) else _ for _ in v
            )
            if k in self.state_dict:
                self.state_dict[k].append(v)
            else:
                self.state_dict[k] = [v]

    @pt.inference_mode()
    def handle(self):
        log_dict = {}
        for okeys, op in zip(self.okeys, self.ops):
            for okey in okeys:
                metric, valid = list(zip(*self.state_dict[okey]))
                metric = np.concatenate(metric, 0)  # concat all batches  # (b,..)
                valid = np.concatenate(valid, 0)  # (b,)
                metric2 = metric[valid]  # keep the valid ones  # (?,..)
                v2 = np.__dict__[op](metric2, 0)
                log_dict[okey] = v2.tolist()
        if self.log_file:
            __class__.save(self.idx, log_dict, self.log_file)
        print(self.idx, log_dict)
        return log_dict

    @staticmethod
    @pt.inference_mode()
    def save(key, avg_dict, log_file):
        line = json.dumps({key: avg_dict})
        with open(log_file, "a") as f:
            f.write(line + "\n")

    @pt.inference_mode()
    def before_epoch(self, isval, **pack):
        epoch = pack[self.epoch_key]
        self.index(epoch, isval)
        return pack

    @pt.inference_mode()
    def after_step(self, **pack):
        for ikeys, okeys in zip(self.ikeys, self.okeys):
            for ikey, okey in zip(ikeys, okeys):
                value = DictTool.getattr(pack, ikey)
                self.append({okey: value})
        return pack

    @pt.inference_mode()
    def after_epoch(self, **pack):
        self.handle()
        return pack


class SaveModel(Callback):
    """"""

    def __init__(
        self,
        save_dir=None,
        since_step=0,
        weights_only=True,
        key=r".*",
        epoch_key="epoch",
        epoch_v_key="epoch_v",
        step_count_key="step_count",
        model_key="model",
    ):
        super().__init__()
        self.save_dir = save_dir
        self.since_step = since_step  # self.after_step is taken
        self.weights_only = weights_only
        self.key = key
        self.epoch_key = epoch_key
        self.step_count_key = step_count_key
        self.model_key = model_key

    @pt.inference_mode()
    def __call__(self, epoch, step_count, model):
        if step_count >= self.since_step:
            save_file = Path(self.save_dir) / f"{epoch:04d}.pth"
            model.save(save_file, self.weights_only, self.key)

    @pt.inference_mode()
    def after_epoch(self, **pack):
        epoch = pack[self.epoch_key]
        step_count = pack[self.step_count_key]
        model = pack[self.model_key]
        self.__call__(epoch, step_count, model)
        return pack



class SaveCkpt(Callback):

    def __init__(
        self,
        save_dir=None,
        since_step=0,
        epoch_key="epoch",
        epoch_v_key="epoch_v",
        step_count_key="step_count",
        val_interval_key="val_interval",
        total_step_key="total_step",
        model_key="model",
        optimiz_key="optimiz",
    ):
        super().__init__()
        self.save_dir = save_dir
        self.since_step = since_step
        self.epoch_key = epoch_key
        self.epoch_v_key = epoch_v_key
        self.step_count_key = step_count_key
        self.val_interval_key = val_interval_key
        self.total_step_key = total_step_key
        self.model_key = model_key
        self.optimiz_key = optimiz_key


    @pt.inference_mode()
    def __call__(self, epoch, epoch_v, step_count, model, optimiz):

        if step_count >= self.since_step:
            save_file = Path(self.save_dir) / f"{epoch:04d}.pth"

            ckpt = {
                "epoch": epoch,
                "epoch_v": epoch_v,
                "step_count": step_count,
                "model": model.state_dict(),
                "optimiz": optimiz.state_dict(),
                "gscale": optimiz.gscale.state_dict(),
            }
            # ckpt 无需 scheduler_state_dict, 因为老师自定义的 scheduler 是一个根据 step 预先计算好的模块, 我们只需要存储 step_count 即可


            pt.save(ckpt, save_file)


    @pt.inference_mode()
    def after_epoch(self, **pack):

        epoch = pack[self.epoch_key]
        epoch_v = pack[self.epoch_v_key]
        val_interval = pack[self.val_interval_key]
        step_count = pack[self.step_count_key]
        total_step = pack[self.total_step_key]

        flag1 = step_count >= (epoch_v + 1) * val_interval
        flag2 = step_count >= total_step
        if flag1 or flag2:
            self.__call__(
                epoch=epoch,
                epoch_v=epoch_v,
                step_count=step_count,
                model=pack[self.model_key],
                optimiz=pack[self.optimiz_key],
            )

        return pack



class ExtractSlots(Callback):

    def __init__(
        self,
        save_dir="../datasets/clevrer",
        split_key="split",
        slots_key="output.slotz",
        extra_keys=[],  # ["output.feature", "output.attent2", "output.recon"]
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.split_key = split_key
        self.slots_key = slots_key
        self.extra_keys = extra_keys

    @pt.inference_mode()
    def before_epoch(self, **pack):
        split = pack[self.split_key]
        assert split in ["train", "val", "test"]

        self.save_dir.mkdir(parents=True, exist_ok=True)
        lmdb_file = self.save_dir / f"slotz_{split}.lmdb"
        self.lmdb_env = lmdb_open_write(lmdb_file)

        self.cnt = 0
        self.keys = []

        return pack


    @pt.inference_mode()
    def after_step(self, **pack):

        slotz = DictTool.getattr(pack, self.slots_key)
        slotz = slotz.detach().cpu().numpy()
        b = slotz.shape[0]

        extra_data = {}

        for k in self.extra_keys:
            v = DictTool.getattr(pack, k)
            v = v.detach().cpu().numpy()
            extra_data[k.split('.')[-1]] = v

        txn = self.lmdb_env.begin(write=True)
        for i in range(b):
            sample_key = f"{self.cnt:06d}".encode("ascii")
            self.keys.append(sample_key)
            sample_dict = dict(
                slotz=slotz[i],  # (t,n,c)
            )
            for k, v in extra_data.items():
                sample_dict[k] = v[i]
            txn.put(sample_key, pkl.dumps(sample_dict))
            self.cnt += 1
        txn.commit()

        return pack


    @pt.inference_mode()
    def after_epoch(self, **pack):
        txn = self.lmdb_env.begin(write=True)
        txn.put(b"__keys__", pkl.dumps(self.keys))
        txn.commit()

        self.lmdb_env.close()

        return pack






