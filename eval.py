from argparse import ArgumentParser
from pathlib import Path

from einops import rearrange
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import tqdm

from object_centric_bench.datum import DataLoader
from object_centric_bench.util_datum import draw_segmentation_np
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


@pt.inference_mode()
def val_epoch(cfg, dataset_v, model, loss_fn, acc_fn_v, callback_v, vis=False):
    pack = Config({})
    pack.dataset_v = dataset_v
    pack.model = model
    pack.loss_fn = loss_fn
    pack.acc_fn_v = acc_fn_v
    pack.callback_v = callback_v
    pack.epoch = 0

    mean = pt.from_numpy(np.array(cfg.IMAGENET_MEAN, "float32"))
    std = pt.from_numpy(np.array(cfg.IMAGENET_STD, "float32"))

    pack.isval = True
    pack.model.eval()
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for i, batch in enumerate(tqdm.tqdm(pack.dataset_v)):
        pack.batch = batch

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn(**pack)
        pack.acc = pack.acc_fn_v(**pack)

        if vis and "segment2" in pack.output:
            imgs = ((pack.batch["video"] * std.cuda() + mean.cuda()).clip(0, 255).byte())
            segs = pack.output["segment2"]

            for img, seg in zip(imgs, segs):
                img = rearrange(img, 't c h w -> t h w c').cpu().numpy()
                seg = seg.cpu().numpy()
                pack.dataset_v.dataset.visualiz(img, segment=seg, wait=0)

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]


def main():
    parser = ArgumentParser(
        description="Evaluate a trained SAVi or Aloe model on the validation set."
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Path to the config file (e.g. config-savi/savi-clevrer_video.py or config-aloe/aloe-clevrer_vqa_slots.py).",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        required=True,
        help="Path to the trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets",
        help="Path to the datasets directory.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Enable visualization of segmentation results (SAVi only).",
    )
    args = parser.parse_args()

    pt.backends.cudnn.benchmark = True

    cfg_file = Path(args.cfg_file)
    data_path = Path(args.data_dir)
    ckpt_file = Path(args.ckpt_file)

    assert cfg_file.name.endswith(".py")
    assert cfg_file.is_file()
    cfg_name = cfg_file.name.split(".")[0]
    cfg = Config.fromfile(cfg_file)
    cfg.name = cfg_name

    # datum init
    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path

    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )

    # model init
    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    ckpt = pt.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if cfg.freez:
        model.freez(cfg.freez, verbose=False)

    model = model.cuda()

    # learn init
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))

    cfg.callback_v = [_ for _ in cfg.callback_v if _.type.__name__ != "SaveCkpt"]
    for cb in cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = None
    callback_v = build_from_config(cfg.callback_v)

    # do eval
    val_epoch(cfg, dataload_v, model, loss_fn, acc_fn_v, callback_v, vis=args.vis)


if __name__ == "__main__":
    main()