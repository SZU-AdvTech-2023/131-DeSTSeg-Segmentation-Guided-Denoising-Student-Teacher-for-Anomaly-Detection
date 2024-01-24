import argparse
import os
import shutil
import warnings
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from eval import evaluate
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss,smoothL1_loss

warnings.filterwarnings("ignore")


def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}_mem"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.se_att.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    # scheduler1 = lr_scheduler.MultiStepLR(seg_optimizer, milestones=[300,900], gamma=0.1)
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
            {"params": model.memory[0].parameters(), "lr": args.lr_mem},
            {"params": model.memory[1].parameters(), "lr": args.lr_mem},
            {"params": model.memory[2].parameters(), "lr": args.lr_mem},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    # mem_optimizer = torch.optim.SGD(
    #     [
            # {"params": model.memory[0].parameters(), "lr": args.lr_mem},
            # {"params": model.memory[1].parameters(), "lr": args.lr_mem},
            # {"params": model.memory[2].parameters(), "lr": args.lr_mem},
    #     ],
    #     lr=0.4,
    #     momentum=0.9,
    #     weight_decay=1e-4,
    #     nesterov=False,
    # )
    # scheduler2 = lr_scheduler.MultiStepLR(de_st_optimizer, milestones=[300,900], gamma=0.1)
    dataset = MVTecDataset(
        is_train=True,
        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        normalize_mean=NORMALIZE_MEAN,
        normalize_std=NORMALIZE_STD,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,pin_memory=True
    )

    global_step = 0

    flag = True

    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            # mem_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()
            mask_origin = sample_batched["mask_normal"].cuda()

            if global_step < args.de_st_steps:
                model.student_net.train()
                model.segmentation_net.eval()
            else:
                model.student_net.eval()
                model.segmentation_net.train()

            output_segmentation,output_de_st_list,st_loss = model(
                img_aug, img_origin
            )

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )
            mask_origin = F.interpolate(
                mask_origin,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask_origin = torch.where(
                mask_origin < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            st_loss_val = cosine_similarity_loss(st_loss)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask) 

            if global_step < args.de_st_steps:
                total_loss_val = cosine_loss_val + st_loss_val*0.1
                total_loss_val.backward()
                de_st_optimizer.step()
                # mem_optimizer.step()
                # scheduler2.step()
            else:
                total_loss_val = focal_loss_val + l1_loss_val
                total_loss_val.backward()
                seg_optimizer.step()
                # scheduler1.step()

            global_step += 1

            visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            visualizer.add_scalar("mem_loss", st_loss_val, global_step)
            visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            visualizer.add_scalar("total_loss", total_loss_val, global_step)

            if global_step % args.eval_per_steps == 0 and global_step >= 1000:
                evaluate(args, category, model, visualizer, global_step)

            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"Training at global step {global_step}, cosine loss: {round(float(cosine_loss_val), 4)}, mem loss: {round(float(st_loss_val), 4)}"
                    )
                else:
                    print(
                        f"Training at global step {global_step}, focal loss: {round(float(focal_loss_val), 4)}, l1 loss: {round(float(l1_loss_val), 4)}"
                    )

            if global_step >= args.steps:
                flag = False
                break

    torch.save(
        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32)

    parser.add_argument("--mvtec_path", type=str, default="/home/chengjia/dataset/chengjia/mvtec_AD/")
    parser.add_argument("--dtd_path", type=str, default="./scripts/datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model_att/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs_att/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_mem", type=float, default=0.04)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=1000)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument(
        "--custom_training_category", action="store_true", default=False
    )
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--heatmap_path", type=str, default="./heatmap_att/")
    parser.add_argument("--category", nargs="*", type=str, default=["screw"])
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args()

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        # check
        for category in (
            no_rotation_category + slight_rotation_category + rotation_category
        ):
            assert category in ALL_CATEGORY
    else:
        no_rotation_category = [
            "capsule",
            "metal_nut",
            "pill",
            "toothbrush",
            "transistor",
        ]
        slight_rotation_category = [
            "wood",
            "zipper",
            "cable",
        ]
        rotation_category = [
            "bottle",
            "grid",
            "hazelnut",
            "leather",
            "tile",
            "carpet",
            "screw",
        ]

    with torch.cuda.device(args.gpu_id):
        for obj in no_rotation_category:
            print(obj)
            train(args, obj)

        for obj in slight_rotation_category:
            print(obj)
            train(args, obj, rotate_90=False, random_rotate=5)

        for obj in rotation_category:
            print(obj)
            train(args, obj, rotate_90=True, random_rotate=5)
