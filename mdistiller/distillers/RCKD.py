import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def rckd_loss(selected_log_p_student, selected_log_p_teacher, temperature):

    student_log_ratios = selected_log_p_student.unsqueeze(2) - selected_log_p_student.unsqueeze(1)
    teacher_log_ratios = selected_log_p_teacher.unsqueeze(2) - selected_log_p_teacher.unsqueeze(1)

    # 提取上三角部分（不包括对角线）
    triu_mask = torch.triu(torch.ones_like(student_log_ratios[0]), diagonal=1).bool()
    student_vec = student_log_ratios[:, triu_mask]
    teacher_vec = teacher_log_ratios[:, triu_mask]

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(student_vec, teacher_vec, dim=1)
    loss_cos = 1 - cos_sim.mean()
    loss_cos *= temperature ** 2
    return loss_cos


class RCKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(RCKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        
        log_p_student = F.log_softmax(logits_student / self.temperature, dim=1)
        log_p_teacher = F.log_softmax(logits_teacher / self.temperature, dim=1)
        
        
        loss_kd = self.kd_loss_weight * rckd_loss(log_p_student, log_p_teacher, self.temperature)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
