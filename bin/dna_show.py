from datetime import timedelta
from timeit import default_timer as timer

from omegaconf import OmegaConf

import dna
from dna.camera import ImageProcessor, ImageCaptureType, image_capture_type, load_image_capture
from dna.platform import DNAPlatform

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Display a video")
    parser.add_argument("camera", metavar='camera_uri', help="target camera uri")
    parser.add_argument("--conf", help="DNA framework configuration", default=dna.DNA_CONIFIG_FILE)
    parser.add_argument("--no_sync", help="no sync to fps", action="store_true")
    parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()

    uri = args.camera
    cap_type = image_capture_type(uri)
    if cap_type == ImageCaptureType.PLATFORM:
        conf = OmegaConf.load(args.conf)
        platform = DNAPlatform.load_from_config(conf.platform)
        _, camera_info = platform.get_resource("camera_infos", (uri,))
        uri = camera_info.uri
    cap = load_image_capture(uri, sync=not args.no_sync, begin_frame=args.begin_frame, end_frame=args.end_frame)

    with ImageProcessor(cap, window_name="output") as processor:
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={processor.fps_measured:.1f}" )