import os.path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
import warnings


class Trainer:
    def __init__(self, net, train_dataset, test_dataset, args, conf, device=None):
        self.args = args
        self.net = net # 赋值pixelnerf网络
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, # batch_size 4，一次获得4个场景的图片
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        ) # todo 这里numworkers默认是8，调试完毕修改
        self.test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=min(args.batch_size, 16),
            shuffle=True,
            num_workers=4,
            pin_memory=False,
        )

        self.num_total_batches = len(self.train_dataset)
        self.exp_name = args.name
        self.save_interval = conf.get_int("save_interval") # 每隔50iter保存一次
        self.print_interval = conf.get_int("print_interval") # 每隔2个iter保存一次
        self.vis_interval = conf.get_int("vis_interval") # 每隔100个iter可视化一次？
        self.eval_interval = conf.get_int("eval_interval") # 每隔50个iter就验证一次
        self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1) # 不重复epoch
        self.num_epochs = args.epochs # 训练10000万个epoch
        self.accu_grad = conf.get_int("accu_grad", 1)
        self.summary_path = os.path.join(args.logs_path, args.name)
        self.writer = SummaryWriter(self.summary_path)

        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test

        os.makedirs(self.summary_path, exist_ok=True)

        # Currently only Adam supported
        self.optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        if args.gamma != 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optim, gamma=args.gamma
            )
        else:
            self.lr_scheduler = None

        # Load weights
        self.managed_weight_saving = hasattr(net, "load_weights") # 是否要读取权重
        if self.managed_weight_saving:
            net.load_weights(self.args)
        self.iter_state_path = "%s/%s/_iter" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.optim_state_path = "%s/%s/_optim" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.lrsched_state_path = "%s/%s/_lrsched" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.default_net_state_path = "%s/%s/net" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.start_iter_id = 0
        if args.resume: # 如果要求读取权重
            if os.path.exists(self.optim_state_path):
                try:
                    self.optim.load_state_dict(
                        torch.load(self.optim_state_path, map_location=device)
                    )
                except:
                    warnings.warn(
                        "Failed to load optimizer state at", self.optim_state_path
                    )
            if self.lr_scheduler is not None and os.path.exists(
                self.lrsched_state_path
            ):
                self.lr_scheduler.load_state_dict(
                    torch.load(self.lrsched_state_path, map_location=device)
                )
            if os.path.exists(self.iter_state_path):
                self.start_iter_id = torch.load(
                    self.iter_state_path, map_location=device
                )["iter"]
            if not self.managed_weight_saving and os.path.exists(
                self.default_net_state_path
            ):
                net.load_state_dict(
                    torch.load(self.default_net_state_path, map_location=device)
                )

        self.visual_path = os.path.join(self.args.visual_path, self.args.name)
        self.conf = conf

    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def train_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None

    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + str(losses[k]) for k in losses))

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        test_data_iter = data_loop(self.test_data_loader)

        step_id = self.start_iter_id

        progress = tqdm.tqdm(bar_format="[{rate_fmt}] ")
        for epoch in range(self.num_epochs): # 一个epoch跑一遍所有场景
            self.writer.add_scalar(
                "lr", self.optim.param_groups[0]["lr"], global_step=step_id
            ) # 日志保存lr信息

            batch = 0
            for _ in range(self.num_epoch_repeats):
                for data in self.train_data_loader: # 读取一个batch的场景信息，注意data的img_id其实是指的场景的id而不是图片
                    losses = self.train_step(data, global_step=step_id) # 对场景进行训练，注意并行数量batch_size
                    loss_str = fmt_loss_str(losses)
                    if batch % self.print_interval == 0: # 每完成二次Batch的训练就输出结果
                        print(
                            "E",
                            epoch,
                            "B",
                            batch,
                            loss_str,
                            " lr",
                            self.optim.param_groups[0]["lr"],
                        )

                    if batch % self.eval_interval == 0: # 每完成50次Batch就进行评估
                        test_data = next(test_data_iter) # 随机选择4个场景
                        self.net.eval()
                        with torch.no_grad(): # 这里和train的区别就是loss不做反向传播，直接打印结果
                            test_losses = self.eval_step(test_data, global_step=step_id)
                        self.net.train()
                        test_loss_str = fmt_loss_str(test_losses)
                        self.writer.add_scalars("train", losses, global_step=step_id)
                        self.writer.add_scalars(
                            "test", test_losses, global_step=step_id
                        )
                        print("*** Eval:", "E", epoch, "B", batch, test_loss_str, " lr")

                    if batch % self.save_interval == 0 and (epoch > 0 or batch > 0):
                        print("saving")
                        if self.managed_weight_saving:
                            self.net.save_weights(self.args)
                        else:
                            torch.save(
                                self.net.state_dict(), self.default_net_state_path
                            )
                        torch.save(self.optim.state_dict(), self.optim_state_path)
                        if self.lr_scheduler is not None:
                            torch.save(
                                self.lr_scheduler.state_dict(), self.lrsched_state_path
                            )
                        torch.save({"iter": step_id + 1}, self.iter_state_path)
                        self.extra_save_state()

                    if batch % self.vis_interval == 0: # 每训练100个batch就可视化一次
                        print("generating visualization")
                        if self.fixed_test:
                            test_data = next(iter(self.test_data_loader))
                        else:
                            test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            vis, vis_vals = self.vis_step(
                                test_data, global_step=step_id
                            )
                        if vis_vals is not None:
                            self.writer.add_scalars(
                                "vis", vis_vals, global_step=step_id
                            )
                        self.net.train()
                        if vis is not None:
                            import imageio

                            vis_u8 = (vis * 255).astype(np.uint8)
                            imageio.imwrite( # 上面一行是粗网络，下面一行是细网络
                                os.path.join(
                                    self.visual_path,
                                    "{:04}_{:04}_vis.png".format(epoch, batch),
                                ),
                                vis_u8,
                            )

                    if (
                        batch == self.num_total_batches - 1
                        or batch % self.accu_grad == self.accu_grad - 1 # 最后的这一行表达式好像一直为真
                    ):
                        self.optim.step()
                        self.optim.zero_grad() # 清空梯度

                    self.post_batch(epoch, batch)
                    step_id += 1 # 更新batch数目
                    batch += 1
                    progress.update(1)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
