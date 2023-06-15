"""
    Helper class to create and add images to video
"""
import imageio
import numpy as np

class Video():
    def __init__(self, path, name='video_log.mp4', mode='I', fps=30, codec='libx264', bitrate='16M') -> None:
        
        if path[-1] != "/":
            path += "/"
            
        self.writer = imageio.get_writer(path+name, mode=mode, fps=fps, codec=codec, bitrate=bitrate)
    
    def ready_image(self, image, write_video=True):
        # assuming channels last - as renderer returns it
        if len(image.shape) == 4: 
            image = image.squeeze(0)[..., :3].detach().cpu().numpy()
        else:
            image = image[..., :3].detach().cpu().numpy()

        image = np.clip(np.rint(image*255.0), 0, 255).astype(np.uint8)

        if write_video:
            self.writer.append_data(image)

        return image

    def close(self):
        self.writer.close()