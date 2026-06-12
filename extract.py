from argparse import ArgumentParser
from pathlib import Path

from einops import rearrange
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch as pt

from object_centric_bench.datum import DataLoader
from object_centric_bench.util_datum import draw_segmentation_np
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


@pt.inference_mode()
def extract_epoch(split, dataset, dataloader, model, loss_fn, acc_fn_v, callback_extract):
    pack = Config({})
    pack.split = split
    pack.dataloader = dataloader
    pack.model = model
    pack.loss_fn = loss_fn
    pack.acc_fn_v = acc_fn_v
    pack.callback_extract = callback_extract
    pack.epoch = 0

    pack.isval = True
    pack.model.eval()

    print(f"slots_{split} is extracting")
    [_.before_epoch(**pack) for _ in pack.callback_extract]

    for i, batch in enumerate(pack.dataloader):
        pack.batch = batch

        [_.before_step(**pack) for _ in pack.callback_extract]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)

            [_.after_forward(**pack) for _ in pack.callback_extract]
            pack.loss = pack.loss_fn(**pack)
        pack.acc = pack.acc_fn_v(**pack)

        [_.after_step(**pack) for _ in pack.callback_extract]

    [_.after_epoch(**pack) for _ in pack.callback_extract]


def main():
    parser = ArgumentParser(
        description="Extract object-centric slots from a trained SAVi checkpoint."
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="archive-savi/savi-clevrer_video/savi-clevrer_video.py",
        help="Path to the config file used for SAVi training.",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="archive-savi/savi-clevrer_video/42_ckpt/0011.pth",
        help="Path to the trained SAVi checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets",
        help="Path to the datasets directory.",
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
    transform_v = cfg.dataset_v.transform
    split_map = dict(train="dataset_t",
                     val="dataset_v",
                     test="dataset_test")

    split = []
    dataset = []
    dataloader = []
    for s in ["train", "val", "test"]:
        d_name = split_map[s]
        d_cfg = getattr(cfg, d_name, None)

        if d_cfg is not None:
            d_cfg.base_dir = data_path
            d_cfg.transform = copy.deepcopy(transform_v)
            d_cfg.sample_mode = "extract"

            d = build_from_config(d_cfg)
            dl = DataLoader(
                d,
                cfg.batch_size_v,
                shuffle=False,
                num_workers=cfg.num_work,
                collate_fn=build_from_config(cfg.collate_fn_v),
                pin_memory=True,
            )

            split.append(s)
            dataset.append(d)
            dataloader.append(dl)

    # model init
    model = build_from_config(cfg.model)
    print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if ckpt_file:
        if isinstance(ckpt_file, (list, tuple)):
            assert len(ckpt_file) == 1
            ckpt_file = ckpt_file[0]
        ckpt = pt.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])

    if cfg.freez:
        model.freez(cfg.freez, verbose=False)
    model = model.cuda()

    # learn init
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))

    cfg.callback_extract = [_ for _ in cfg.callback_extract if _.type.__name__ != "SaveCkpt"]
    for cb in cfg.callback_extract:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = None
    callback_extract = build_from_config(cfg.callback_extract)

    # do eval
    for s, d, dl in zip(split, dataset, dataloader):
        extract_epoch(s, d, dl, model, loss_fn, acc_fn_v, callback_extract)


if __name__ == "__main__":
    main()