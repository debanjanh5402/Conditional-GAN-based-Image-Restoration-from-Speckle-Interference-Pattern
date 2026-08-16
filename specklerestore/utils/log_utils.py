from tqdm import tqdm


def _log_epoch_summary_unet(epoch:int, train_history:dict, val_history:dict|None = None):
    tqdm.write(f"{'-'*35} Epoch {epoch} Summary {'-'*35}")
    tqdm.write(f"train: loss={train_history['loss']:0.4f}, ssim={train_history['ssim']:0.4f}, psnr={train_history['psnr']:0.4f}")
    if val_history is not None:
        tqdm.write(f"  val: loss={val_history['loss']:0.4f}, ssim={val_history['ssim']:0.4f}, psnr={val_history['psnr']:0.4f}")


def _log_epoch_summary_pix2pix(epoch:int, train_history: dict, val_history:dict|None=None):
     tqdm.write(f"{'-'*35} Epoch {epoch} Summary {'-'*35}")
     tqdm.write(f"train: g_loss={train_history['g_loss']:0.4f}, g_adv_loss={train_history['g_adv_loss']:0.4f}, g_recon_loss={train_history['g_recon_loss']:0.4f}")
     tqdm.write(f"       d_loss={train_history['d_loss']:0.4f}, d_real_loss={train_history['d_real_loss']:0.4f}, d_fake_loss={train_history['d_fake_loss']:0.4f}")
     tqdm.write(f"       ssim={train_history['ssim']:0.6f}, psnr={train_history['psnr']:0.4f}")

     if val_history is not None:
          tqdm.write(f"  val: g_loss={val_history['g_loss']:0.4f}, g_adv_loss={val_history['g_adv_loss']:0.4f}, g_recon_loss={val_history['g_recon_loss']:0.4f}")
          tqdm.write(f"       d_loss={val_history['d_loss']:0.4f}, d_real_loss={val_history['d_real_loss']:0.4f}, d_fake_loss={val_history['d_fake_loss']:0.4f}")
          tqdm.write(f"       ssim={val_history['ssim']:0.6f}, psnr={val_history['psnr']:0.4f}")

def _log_step_summary_pix2pix(step: int, train_history: dict, val_history: dict | None = None):
    tqdm.write(f"{'-'*35} Step {step} Summary {'-'*35}")
    tqdm.write(f"train: g_loss={train_history.get('g_loss', 0):0.4f}, d_loss={train_history.get('d_loss', 0):0.4f}")
    
    if val_history is not None:
        tqdm.write(f"  val: g_loss={val_history.get('g_loss', 0):0.4f}, d_loss={val_history.get('d_loss', 0):0.4f}")
        tqdm.write(f"       ssim={val_history.get('ssim', 0):0.6f}, psnr={val_history.get('psnr', 0):0.4f}")