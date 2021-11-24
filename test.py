"""
测试程序
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/hoshino/CLionProjects/copy_sjtu_radar/resources/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='/home/hoshino/CLionProjects/copy_sjtu_radar/resources/demo_pic.jpg',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2],
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    save_img = True
    save_txt = False
    vid_path, vid_writer = None, None
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    dataset = LoadImages('/home/hoshino/CLionProjects/copy_sjtu_radar/resources/two_cam/1.mp4', img_size=640)
    predictor = Predictor()
    for path, img, im0s, vid_cap in dataset:
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)
        save_path = str(save_dir / p.name)
        t1 = time_synchronized()
        output = predictor.detect_cars(im0s)
        t2 = time_synchronized()
        print(f'{s}Done. ({t2 - t1:.3f}s)')
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, output)
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(output)

