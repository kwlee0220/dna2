from datetime import timedelta
from timeit import default_timer as timer

from dna import DNA_CONIFIG_FILE, parse_config_args, load_config
from dna.camera import Camera, ImageProcessor
from dna.platform import DNAPlatform


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Display a video")
    parser.add_argument("node", metavar='node_id', help="target node_id")
    parser.add_argument("--conf", help="DNA framework configuration", default=DNA_CONIFIG_FILE)
    parser.add_argument("--no_sync", help="no sync to fps", action="store_true")
    parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()
    config_grp = parse_config_args(unknown)

    conf = load_config(DNA_CONIFIG_FILE, args.node)
    camera_info = Camera.from_conf(conf.camera)
    cap = camera_info.get_capture(sync=not args.no_sync, begin_frame=args.begin_frame, end_frame=args.end_frame)

    with ImageProcessor(cap, window_name=args.node) as processor:
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={processor.fps_measured:.1f}" )