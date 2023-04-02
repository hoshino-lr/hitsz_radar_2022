import sys
from typing import Optional, Callable

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from loguru import logger

import config
from abstraction.pipeline import ProcessPipeline
from config_type import VideoConfig, HikCameraConfig, RdrConfig, RdrReceive
from service.video_service import VideoReader
from service.abstract_service import StartStoppableTrait
from service.camera_service import HikCameraService
from service.rdr_service import RdrThread
# from service.process_thread import ProcessService
from ui.mainwindow.mainwindow import MainWindow

if __name__ == "__main__":
    import time
    logger_name = time.strftime('%Y-%m-%d %H-%M-%S')
    logger.add(f"logs/{logger_name}.log")

    app = QtWidgets.QApplication(sys.argv)

    threads: list[StartStoppableTrait] = list()

    pipelines = {}
    main_reader_service: Optional[VideoReader] = None

    # 遍历 neo_camera_config 字典
    for name, cam_config in config.neo_camera_config.items():
        if not cam_config.enable:
            logger.info(f"相机 {name} 已被标记为禁用，跳过管线初始化")
            continue
        logger.info(f"正在初始化相机 {name} 的管线")

        reader_service: Optional[VideoReader] = None
        rdr_service: Optional[RdrThread] = None
        camera_service: Optional[HikCameraService] = None
        resolution: tuple[int, int] = (3072, 2048)
        camera_fps: Optional[Callable[[], float]] = None
        process_fps: Optional[Callable[[], float]] = None

        match cam_config:
            case VideoConfig(path=path):
                reader_service = VideoReader(cam_config)
                get_camera_provider = reader_service.get_latest_frame_getter
                camera_fps = reader_service.get_fps_getter()
                resolution = reader_service.resolution
                threads.append(reader_service)
            case HikCameraConfig():
                camera_service = HikCameraService(name, cam_config)
                threads.append(camera_service)
                get_camera_provider = camera_service.get_latest_frame_getter
                camera_fps = camera_service.get_fps_getter()
                resolution = cam_config.roi[2:4]
            case RdrConfig():
                rdr_service = RdrThread(cam_config)
                # rdr_thread.start()
                threads.append(rdr_service)
                get_camera_provider = rdr_service.get_latest_frame_getter
                camera_fps = rdr_service.get_fps_getter()
            case _:
                raise ValueError(f"Unknown camera config: {cam_config}")

        match cam_config.net_process:
            case True:
                # 暂时忽略从录制读取框框，当作完整的流程
                from service.process_service import ProcessService

                process = ProcessService(resolution, get_camera_provider(), name)
                # process = ProcessService((3072, 2048), get_camera_provider())
                threads.append(process)
                get_process_provider = process.get_net_data_provider
                process_fps = process.get_fps_getter()
            case False:
                get_process_provider = lambda: lambda: None
                process_fps = lambda: 0.0
            case str():
                get_process_provider = reader_service.get_latest_armor_getter()
                process_fps = camera_fps
            case RdrReceive():
                get_process_provider = rdr_service.get_latest_armor_getter()
                process_fps = camera_fps
            case _:
                raise ValueError(f"Unknown net_process config: {cam_config.net_process}")

        if name == "cam_left":
            main_reader_service = reader_service
        providers_for_ui = ProcessPipeline(cam_config, get_camera_provider(), get_process_provider(), camera_fps,
                                           process_fps)
        pipelines[name] = providers_for_ui
    for th in threads:
        th.start()
    logger.debug("即将构造 MainWindow")
    window = MainWindow(pipelines, main_reader_service)
    logger.info("完成构造 MainWindow")
    window.show()
    timer_main = QTimer()
    timer_main.timeout.connect(window.spin)
    timer_main.start(0)

    ret = app.exec_()
    logger.info("GUI 退出，正在等待线程退出")
    for th in threads:
        th.stop()
    logger.info("所有线程退出")
    sys.exit(ret)
