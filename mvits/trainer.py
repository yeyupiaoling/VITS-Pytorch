import json
import os

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualdl import LogWriter

from mvits.data_utils.collate_fn import TextAudioSpeakerCollate
from mvits.data_utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from mvits.data_utils.reader import TextAudioSpeakerLoader
from mvits.data_utils.sampler import DistributedBucketSampler
from mvits.models import commons
from mvits.models.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mvits.models.models import SynthesizerTrn, MultiPeriodDiscriminator
from mvits.utils.logger import setup_logger
from mvits.utils.utils import load_checkpoint, save_checkpoint, plot_spectrogram_to_numpy, dict_to_object

logger = setup_logger(__name__)


class VITSTrainer(object):
    def __init__(self, config_path, args):
        assert torch.cuda.is_available(), "CPU training is not allowed."
        os.makedirs(args.model_dir, exist_ok=True)
        config_save_path = os.path.join(args.model_dir, "config.json")
        with open(config_path, "r") as f:
            data = f.read()
        # 复制一份
        with open(config_save_path, "w") as f:
            f.write(data)
        config = json.loads(data)

        # 读取配置文件
        hparams = dict_to_object(config)
        hparams.model_dir = args.model_dir
        hparams.train.epochs = args.epochs
        hparams.drop_speaker_embed = args.drop_speaker_embed
        self.hps = hparams
        self.symbols = self.hps['symbols']

    def __setup_dataloader(self, rank, n_gpus):
        train_dataset = TextAudioSpeakerLoader(self.hps.data.training_files, self.hps.data, self.symbols)
        train_sampler = None
        if n_gpus > 1:
            train_sampler = DistributedBucketSampler(train_dataset,
                                                     self.hps.train.batch_size,
                                                     [32, 300, 400, 500, 600, 700, 800, 900, 1000],
                                                     num_replicas=n_gpus,
                                                     rank=rank,
                                                     shuffle=True)
        collate_fn = TextAudioSpeakerCollate()
        self.train_loader = DataLoader(train_dataset, num_workers=0, shuffle=(train_sampler is None), pin_memory=True,
                                       collate_fn=collate_fn, batch_sampler=train_sampler,
                                       batch_size=self.hps.train.batch_size)
        if rank == 0:
            eval_dataset = TextAudioSpeakerLoader(self.hps.data.validation_files, self.hps.data, self.symbols)
            self.eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
                                          batch_size=self.hps.train.batch_size, pin_memory=True,
                                          drop_last=False, collate_fn=collate_fn)
        logger.info('训练数据：{}'.format(len(train_dataset)))

    def __setup_model(self, rank, n_gpus, resume_model=None, pretrained_model=None):
        latest_epoch = 0
        self.net_g = SynthesizerTrn(len(self.symbols),
                                    self.hps.data.filter_length // 2 + 1,
                                    self.hps.train.segment_size // self.hps.data.hop_length,
                                    n_speakers=self.hps.data.n_speakers,
                                    **self.hps.model).cuda(rank)
        self.net_d = MultiPeriodDiscriminator(self.hps.model.use_spectral_norm).cuda(rank)

        # 自动恢复训练
        latest_g_checkpoint_path = os.path.join(self.hps.model_dir, "G_latest.pth")
        latest_d_checkpoint_path = os.path.join(self.hps.model_dir, "D_latest.pth")
        if os.path.exists(latest_g_checkpoint_path):
            _, _, _, latest_epoch = load_checkpoint(latest_g_checkpoint_path, self.net_g, None)
        if os.path.exists(latest_d_checkpoint_path):
            _, _, _, latest_epoch = load_checkpoint(latest_d_checkpoint_path, self.net_d, None)
        # 加载预训练模型
        if pretrained_model:
            pretrained_g_model_path = os.path.join(pretrained_model, "G_latest.pth")
            pretrained_d_model_path = os.path.join(pretrained_model, "D_latest.pth")
            if os.path.exists(pretrained_g_model_path):
                load_checkpoint(pretrained_g_model_path, self.net_g, None)
            if os.path.exists(pretrained_d_model_path):
                load_checkpoint(pretrained_d_model_path, self.net_d, None)
        # 加载恢复训练模型
        if resume_model:
            resume_g_model_path = os.path.join(resume_model, "G_latest.pth")
            resume_d_model_path = os.path.join(resume_model, "D_latest.pth")
            if os.path.exists(resume_g_model_path):
                load_checkpoint(resume_g_model_path, self.net_g, None)
            if os.path.exists(resume_d_model_path):
                load_checkpoint(resume_d_model_path, self.net_d, None)
        # freeze all other layers except speaker embedding
        for p in self.net_g.parameters():
            p.requires_grad = True
        for p in self.net_d.parameters():
            p.requires_grad = True
        # for p in net_d.parameters():
        #     p.requires_grad = False
        # net_g.emb_g.weight.requires_grad = True
        self.optim_g = torch.optim.AdamW(self.net_g.parameters(),
                                         self.hps.train.learning_rate,
                                         betas=self.hps.train.betas,
                                         eps=self.hps.train.eps)
        self.optim_d = torch.optim.AdamW(self.net_d.parameters(),
                                         self.hps.train.learning_rate,
                                         betas=self.hps.train.betas,
                                         eps=self.hps.train.eps)
        # 多卡
        if n_gpus > 1:
            self.net_g = torch.nn.parallel.DistributedDataParallel(self.net_g, device_ids=[rank])
            self.net_d = torch.nn.parallel.DistributedDataParallel(self.net_d, device_ids=[rank])

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.hps.train.lr_decay)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.hps.train.lr_decay)

        self.amp_scaler = GradScaler(enabled=self.hps.train.fp16_run)
        return latest_epoch

    def train(self, resume_model=None, pretrained_model=None):
        # 获取有多少张显卡训练
        n_gpus = torch.cuda.device_count()
        writer = None
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if n_gpus > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '8000'
            # Use gloo backend on Windows for Pytorch
            dist.init_process_group(backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://',
                                    world_size=n_gpus, rank=rank)
        torch.manual_seed(self.hps.train.seed)
        torch.cuda.set_device(rank)
        if rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        # 获取数据
        self.__setup_dataloader(rank=rank, n_gpus=n_gpus)
        # 获取模型
        latest_epoch = self.__setup_model(rank=rank, n_gpus=n_gpus,
                                          resume_model=resume_model,
                                          pretrained_model=pretrained_model)
        self.train_step = 0
        if rank == 0:
            writer.add_scalar('Train/lr_g', self.scheduler_g.get_last_lr()[0], latest_epoch)
            writer.add_scalar('Train/lr_d', self.scheduler_d.get_last_lr()[0], latest_epoch)
        # 开始训练
        for epoch_id in range(latest_epoch, self.hps.train.epochs):
            epoch_id += 1
            # 训练一个epoch
            self.__train_epoch(rank=rank, writer=writer, epoch=epoch_id)
            self.scheduler_g.step()
            self.scheduler_d.step()
            # 多卡训练只使用一个进程执行评估和保存模型
            if rank == 0:
                writer.add_scalar('Train/lr_g', self.scheduler_g.get_last_lr()[0], epoch_id)
                writer.add_scalar('Train/lr_d', self.scheduler_d.get_last_lr()[0], epoch_id)
                logger.info('=' * 70)
                self.evaluate(generator=self.net_g, eval_loader=self.eval_loader, writer=writer, epoch=epoch_id)
                # 保存模型
                self.__save_model(epoch_id=epoch_id)

    # 保存模型
    def __save_model(self, epoch_id):
        # 保存模型
        save_checkpoint(self.net_g, self.optim_g, self.hps.train.learning_rate, epoch_id,
                        os.path.join(self.hps.model_dir, "G_latest.pth"))
        save_checkpoint(self.net_d, self.optim_d, self.hps.train.learning_rate, epoch_id,
                        os.path.join(self.hps.model_dir, "D_latest.pth"))
        save_checkpoint(self.net_g, self.optim_g, self.hps.train.learning_rate, epoch_id,
                        os.path.join(self.hps.model_dir, f"G_{epoch_id}.pth"))
        save_checkpoint(self.net_d, self.optim_d, self.hps.train.learning_rate, epoch_id,
                        os.path.join(self.hps.model_dir, f"D_{epoch_id}.pth"))
        # 删除旧模型
        old_model_path = os.path.join(self.hps.model_dir, f"G_{epoch_id - 3}.pth")
        if os.path.exists(old_model_path):
            os.remove(old_model_path)
        old_model_path = os.path.join(self.hps.model_dir, f"D_{epoch_id - 3}.pth")
        if os.path.exists(old_model_path):
            os.remove(old_model_path)

    def __train_epoch(self, rank, writer, epoch):
        self.net_g.train()
        self.net_d.train()
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) \
                in enumerate(tqdm(self.train_loader, desc=f'epoch [{epoch}/{self.hps.train.epochs}]')):
            x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
            spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
            y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
            speakers = speakers.cuda(rank, non_blocking=True)

            with autocast(enabled=self.hps.train.fp16_run):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                    (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(x, x_lengths, spec, spec_lengths, speakers)

                mel = spec_to_mel_torch(spec,
                                        self.hps.data.filter_length,
                                        self.hps.data.n_mel_channels,
                                        self.hps.data.sampling_rate,
                                        self.hps.data.mel_fmin,
                                        self.hps.data.mel_fmax)
                y_mel = commons.slice_segments(mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1),
                                                  self.hps.data.filter_length,
                                                  self.hps.data.n_mel_channels,
                                                  self.hps.data.sampling_rate,
                                                  self.hps.data.hop_length,
                                                  self.hps.data.win_length,
                                                  self.hps.data.mel_fmin,
                                                  self.hps.data.mel_fmax)

                y = commons.slice_segments(y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size)

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc
            self.optim_d.zero_grad()
            self.amp_scaler.scale(loss_disc_all).backward()
            self.amp_scaler.unscale_(self.optim_d)
            commons.clip_grad_value_(self.net_d.parameters(), None)
            self.amp_scaler.step(self.optim_d)

            with autocast(enabled=self.hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            self.optim_g.zero_grad()
            self.amp_scaler.scale(loss_gen_all).backward()
            self.amp_scaler.unscale_(self.optim_g)
            grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), None)
            self.amp_scaler.step(self.optim_g)
            self.amp_scaler.update()

            if rank == 0 and batch_idx % self.hps.train.log_interval == 0:
                lr = self.optim_g.param_groups[0]['lr']
                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur,
                                    "loss/g/kl": loss_kl})
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                # 可视化
                for k, v in scalar_dict.items():
                    writer.add_scalar(k, v, self.train_step)
                self.train_step += 1

    def evaluate(self, generator, eval_loader, writer, epoch):
        generator.eval()
        if isinstance(generator, torch.nn.parallel.DistributedDataParallel):
            eval_generator = generator.module
        else:
            eval_generator = generator
        with torch.no_grad():
            for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
                x, x_lengths = x.cuda(0), x_lengths.cuda(0)
                spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
                y, y_lengths = y.cuda(0), y_lengths.cuda(0)
                speakers = speakers.cuda(0)

                # remove else
                x = x[:1]
                x_lengths = x_lengths[:1]
                spec = spec[:1]
                y = y[:1]
                y_lengths = y_lengths[:1]
                speakers = speakers[:1]
                break
            y_hat, attn, mask, *_ = eval_generator.infer(x, x_lengths, speakers, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * self.hps.data.hop_length

            mel = spec_to_mel_torch(spec,
                                    self.hps.data.filter_length,
                                    self.hps.data.n_mel_channels,
                                    self.hps.data.sampling_rate,
                                    self.hps.data.mel_fmin,
                                    self.hps.data.mel_fmax)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(),
                                              self.hps.data.filter_length,
                                              self.hps.data.n_mel_channels,
                                              self.hps.data.sampling_rate,
                                              self.hps.data.hop_length,
                                              self.hps.data.win_length,
                                              self.hps.data.mel_fmin,
                                              self.hps.data.mel_fmax)
        image_dict = {"gen/mel": plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())}
        audio_dict = {"gen/audio": y_hat[0, :, :y_hat_lengths[0]]}
        image_dict.update({"gt/mel": plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})
        # 可视化
        for k, v in image_dict.items():
            writer.add_image(k, v, epoch, dataformats='HWC')
        for k, v in audio_dict.items():
            writer.add_audio(k, v, epoch, self.hps.data.sampling_rate)

        generator.train()
