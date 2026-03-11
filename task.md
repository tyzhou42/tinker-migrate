### Task

**Pipeline地址：**https://github.com/tyzhou42/tinker-migrate



**任务**

将SFT部分的tinker代码迁移到服务器GPU环境上，跑通任务，得到合理结果

包括简单直接对label做的SFT，和带teacher model蒸馏推理的SFT_reasoning pipeline

(蒸馏teacher可以通过在.sh中设置ENABLE_STAGE1="false"来使用历史结果，不需要重新调api)

(使用LoRA)



**环境**

可以在conda.yml文件里查看，服务器目前是CUDA 11.8，按照该环境配置

若因CUDA版本较老产生无法解决问题请告知；目前已和管理员商量升级CUDA

使用环境中的trl-SFTTrainer, peft, transformers等



**服务器**

ssh tzhou3@la3.heinz.cmu.edu

密码：zty516030

6张A6000

/home/fin-llm-finetune

或者可以直接clone到你想用的服务器上



**数据集**：

继续使用ethos hate speech数据集，可以在dataset底下找到划分好的

train大小：~700

val大小: 150

test大小: 150



**日志**：

确保能正常记录先前wandb和其他loggings的结果

（训练正常完成

val loss 正常下降

test accuracy  **60%+**

wandb 有完整曲线)



**提交时间**：

最好周四晚上9点前

最晚周五中午10点前

