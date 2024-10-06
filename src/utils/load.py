from omegaconf import OmegaConf
from transformers import WhisperFeatureExtractor
from src.model.maaca import MAACA
from src.model.maaca_pretrain import MAACAPretrain
from src.processors.alpro_processors import AlproVideoEvalProcessor, AlproVideoTrainProcessor
from src.processors.blip_processors import BlipCaptionProcessor
def load_model_from_config(config_file, cmd_config, is_eval=False, device="cpu", checkpoint=None):

    model_cfg = OmegaConf.load(config_file).model

    if cmd_config.pretraining:
        model = MAACAPretrain.from_config(model_cfg, cmd_config)
    else:
        model = MAACA.from_config(model_cfg, cmd_config)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)

def load_preprocess(config_path):

    config = OmegaConf.load(config_path).preprocess

    vis_processors = dict()
    txt_processors = dict()
    audio_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")
    audio_proc_cfg = config.get("audio_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = AlproVideoTrainProcessor.from_config(vis_train_cfg)
    vis_processors["eval"] = AlproVideoEvalProcessor.from_config(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = BlipCaptionProcessor.from_config(txt_train_cfg)
    txt_processors["eval"] = BlipCaptionProcessor.from_config(txt_eval_cfg)

    if audio_proc_cfg is not None:
        audio_train_cfg = audio_proc_cfg.get("train")
        audio_eval_cfg = audio_proc_cfg.get("eval")
    else:
        audio_train_cfg = None
        audio_eval_cfg = None

    audio_processors["train"] = WhisperFeatureExtractor.from_pretrained(audio_train_cfg.name)
    audio_processors["eval"] = WhisperFeatureExtractor.from_pretrained(audio_eval_cfg.name)

    return vis_processors, txt_processors, audio_processors