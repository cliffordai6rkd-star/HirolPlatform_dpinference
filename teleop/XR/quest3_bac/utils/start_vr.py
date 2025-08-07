import socket
from typing import Text

import netifaces
from pyqrcode import create
from PIL import Image, ImageDraw, ImageFont


def get_local_ip():
    # try to find IP addresses starting with 192.168
    for interface in netifaces.interfaces():
        # Skip loopback interfaces
        if interface.startswith('lo'):
            continue

        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                ip = addr['addr']
                if ip.startswith('192.168.'):
                    return ip

    # fallback to any non-loopback IP
    for interface in netifaces.interfaces():
        if interface.startswith('lo'):
            continue

        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                ip = addr['addr']
                if not ip.startswith('127.'):
                    return ip

    # Last resort
    return '127.0.0.1'


def generate_qr_code(save_path: Text = "launch_qr.png", ip: Text = None) -> None:
    """Generates and saves a QR code that can be scanned to access WebXR page in a VR headset.

    Args:
        save_path (Text): The path where the QR code image will be saved.
        ip (Text, optional): The IP address to be encoded in the QR code. If None, it will use the local IP address.
    """

    if ip is None:
        ip = get_local_ip()
    url = f"https://{ip}:8012?ws=wss://{ip}:8012"
    qr = create(url, error='L', version=5, mode='binary')
    # saves QR code with a reasonable scale
    qr.png(save_path, scale=8, quiet_zone=2)

    # adds the URL text to the image
    img = Image.open(save_path)
    # Create a new image with extra space at the bottom for the URL
    new_img = Image.new('RGB', (img.width, img.height + 40), (255, 255, 255))
    new_img.paste(img, (0, 0))
    img = new_img
    draw = ImageDraw.Draw(img)

    # Add URL text at the bottom of the image
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    # Calculate position for text (centered at bottom)
    text_width = draw.textlength(url, font=font)
    position = ((img.width - text_width) // 2, img.height - 30)

    # Draw text with white background for readability
    text_bbox = draw.textbbox(position, url, font=font)
    draw.rectangle([
        (text_bbox[0] - 5, text_bbox[1] - 5),
        (text_bbox[2] + 5, text_bbox[3] + 5)
    ], fill="white")
    draw.text(position, url, fill="black", font=font)

    # Save the modified image
    img.save(save_path)

    print(f"QR code with URL text saved to {save_path}")


# gets the local ip address of the machine

if __name__ == "__main__":
    generate_qr_code()
