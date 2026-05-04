"""Streamlit GUI for DeepSC image robust transmission demo."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from deepsc_image.channels import ChannelConfig
from deepsc_image.inference import run_inference
from deepsc_image.model import DeepSCImageModel
from deepsc_image.utils import load_checkpoint, load_yaml, pil_to_tensor, resolve_device, tensor_to_pil


@st.cache_resource
def load_model(checkpoint: str | None, semantic_channels: int, base_channels: int, device_name: str) -> tuple[DeepSCImageModel, str]:
    device = resolve_device(device_name)
    model = DeepSCImageModel(semantic_channels=semantic_channels, base_channels=base_channels).to(device)
    if checkpoint:
        try:
            load_checkpoint(checkpoint, model, device)
            status = f"已加载 checkpoint: {checkpoint}"
        except Exception as e:
            status = f"加载 checkpoint 失败: {e}。当前使用随机初始化模型。"
            checkpoint = None
    if not checkpoint:
        status = "未提供 checkpoint：当前使用随机初始化模型，仅用于功能演示；请加载训练权重获得有效重构质量。"
    model.eval()
    return model, status


def main() -> None:
    st.set_page_config(page_title="DeepSC 图像鲁棒传输系统", layout="wide")
    cfg_path = Path("configs/gui.yaml")
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    st.title("基于 DeepSC 的图像鲁棒传输系统")
    st.caption("支持 AWGN/Rayleigh 信道、SNR -5~20 dB、DeepSC 与 JPEG 传统基线对比。")
    with st.sidebar:
        st.header("参数设置")
        uploaded = st.file_uploader("上传本地图像", type=["png", "jpg", "jpeg", "bmp", "webp"])
        ui_cfg = cfg.get("ui", {})
        snr_db = st.slider("SNR (dB)", min_value=int(ui_cfg.get("min_snr_db", -5)), max_value=int(ui_cfg.get("max_snr_db", 20)), value=int(ui_cfg.get("default_snr_db", 10)))
        
        default_channel = str(ui_cfg.get("default_channel", "awgn"))
        if default_channel not in ["none", "awgn", "rayleigh"]:
            default_channel = "awgn"
        channel_type = st.selectbox("信道类型", ["none", "awgn", "rayleigh"], index=["none", "awgn", "rayleigh"].index(default_channel))
        
        image_size = st.number_input("预处理尺寸", min_value=32, max_value=768, value=int(ui_cfg.get("image_size", 256)), step=32)
        jpeg_quality = st.slider("JPEG 基线质量", min_value=5, max_value=95, value=int(cfg.get("baseline", {}).get("jpeg_quality", 35)))
        
        trusted_dir = Path(ui_cfg.get("trusted_checkpoint_dir", "outputs"))
        available_checkpoints = []
        if trusted_dir.exists() and trusted_dir.is_dir():
            available_checkpoints = [str(p) for p in trusted_dir.rglob("*.pth")]
        
        checkpoint = None
        use_manual_checkpoint = st.checkbox("使用非受信任的本地模型路径 (高级/危险)")
        if use_manual_checkpoint:
            st.warning("⚠️ 警告：加载任意路径的 .pth 文件可能导致任意代码执行，请仅用于完全受信任的本地环境。")
            checkpoint = st.text_input("Checkpoint 路径", value=str(cfg.get("checkpoint") or "")) or None
        else:
            options = ["随机初始化模型 (无 Checkpoint)"] + available_checkpoints
            selected = st.selectbox("选择 Checkpoint", options)
            if selected != options[0]:
                checkpoint = selected
    model_cfg = cfg.get("model", {})
    model, status = load_model(checkpoint, int(model_cfg.get("semantic_channels", 32)), int(model_cfg.get("base_channels", 32)), str(cfg.get("device", "auto")))
    st.info(status)
    if uploaded is None:
        st.warning("请在左侧上传一张图像开始演示。")
        return
        
    max_size_mb = float(ui_cfg.get("max_upload_size_mb", 10.0))
    max_pixels = int(ui_cfg.get("max_image_pixels", 4194304))
    
    if uploaded.size > max_size_mb * 1024 * 1024:
        st.error(f"上传失败：图像大小超过限制 ({max_size_mb} MB)。")
        return
        
    try:
        image = Image.open(uploaded)
        image.verify()  # Verify it's a valid image without decoding fully
        
        if image.width * image.height > max_pixels:
            st.error(f"上传失败：图像分辨率超过限制 ({max_pixels} 像素)。")
            return
            
        # Re-open for actual use
        uploaded.seek(0)
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, SyntaxError, OSError) as e:
        st.error("上传失败：文件已损坏或不是受支持的图像格式。")
        return
        
    tensor = pil_to_tensor(image, image_size=int(image_size))
    result = run_inference(model, tensor, ChannelConfig(channel_type, float(snr_db)), jpeg_quality=jpeg_quality)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("发送原图")
        st.image(tensor_to_pil(tensor), use_container_width=True)
    with col2:
        st.subheader("DeepSC 重构图")
        st.image(tensor_to_pil(result.reconstruction), use_container_width=True)
        st.metric("PSNR", f"{result.deepsc_metrics['psnr']:.2f} dB")
        st.metric("SSIM", f"{result.deepsc_metrics['ssim']:.4f}")
        st.metric("端到端延迟", f"{result.latency_ms:.1f} ms")
    with col3:
        st.subheader("传统 JPEG+信道基线")
        st.image(tensor_to_pil(result.baseline), use_container_width=True)
        st.metric("PSNR", f"{result.baseline_metrics['psnr']:.2f} dB")
        st.metric("SSIM", f"{result.baseline_metrics['ssim']:.4f}")
    st.markdown("""
    **说明**：随机初始化模型只能验证软件链路是否通畅，不能代表 DeepSC 性能。请通过训练 CLI 生成 checkpoint 后再进行正式性能对比。
    """)


if __name__ == "__main__":
    main()
