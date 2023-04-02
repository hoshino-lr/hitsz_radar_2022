# `pyrdr` 库以及录像的同步问题

## 问题

- 如果使用 `pyrdr` 库，则在启用网络的时候采用 `class ImageAndArmorClient` 进行相机图像与网络数据同步传输，不启用时采用 `class ImageClient` 进行相机图像传输。
  - 导致问题：图像与网络数据耦合，需要对是否启用网络分类处理

- 当前主程序判断相机类型与网络启用使用的逻辑是
```python
    match cam_config:
        case VideoConfig(path=path):
            reader = RecordReadManager(cam_config)
            get_camera_provider = reader.get_latest_frame_getter
            camera_fps = reader.get_fps_getter()
        case HikCameraConfig():
            camera = HikCameraThread(name, cam_config)
            camera.start()
            get_camera_provider = camera.get_latest_frame_getter
            camera_fps = camera.get_fps_getter()
            reader = None
        case _:
            raise ValueError(f"Unknown camera_service config: {cam_config}")

    match cam_config.net_process:
        case True | str():
            # 暂时忽略从录制读取框框，当作完整的流程
            process = ProcessThread(cam_config.camera_id, cam_config.roi[2:4], get_camera_provider())
            get_process_provider = process.get_net_data_provider
            process_fps = process.get_fps_getter()
        case False:
            get_process_provider = lambda: lambda: None
            process_fps = lambda: 0.0
        case _:
            raise ValueError(f"Unknown net_process config: {cam_config.net_process}")
```
  - 导致问题：没有办法干净又卫生地将图像数据与网络数据分开处理，必须要写一个既可能提供一个又可能提供两个 Provider 的独立的类 `RdrCameraThread`，很丑，上述的逻辑就要改成
```python
    match cam_config:
        case VideoConfig(path=path):
            reader = RecordReadManager(cam_config)
            get_camera_provider = reader.get_latest_frame_getter
            camera_fps = reader.get_fps_getter()
        case HikCameraConfig():
            camera = HikCameraThread(name, cam_config)
            camera.start()
            get_camera_provider = camera.get_latest_frame_getter
            camera_fps = camera.get_fps_getter()
            reader = None
        case RdrConfig():
            camera = RdrCameraThread(cam_config)                  # 而且这个类里面还要判断是否有网络，再选择类型，把两种情况混合，抽象意义不明
            camera.start()
            get_camera_provider = camera.get_latest_frame_getter
            camera_fps = camera.get_fps_getter()
            reader = None
        case _:
            raise ValueError(f"Unknown camera_service config: {cam_config}")

    match cam_config.net_process:
        case True | str():
            # 暂时忽略从录制读取框框，当作完整的流程
            process = ProcessThread(cam_config.camera_id, cam_config.roi[2:4], get_camera_provider())
            get_process_provider = process.get_net_data_provider
            process_fps = process.get_fps_getter()
        case False:
            get_process_provider = lambda: lambda: None
            process_fps = lambda: 0.0
        case RdrReceive():
            # 写不了了，因为这里的网络数据要从前面 Camera 判断里的局部变量 camera_service 里读取
        case _:
            raise ValueError(f"Unknown net_process config: {cam_config.net_process}")
```
由于跑不通，只能把 `RdrCameraThread` 拉出来，改写成 `RdrReceiver`

```python
    if cam_config is RdrConfig:
    rdr_receiver = RdrReceiver(cam_config)
    rdr_receiver.start()
match cam_config:
    case VideoConfig(path=path):
        reader = RecordReadManager(cam_config)
        get_camera_provider = reader.get_latest_frame_getter
        camera_fps = reader.get_fps_getter()
    case HikCameraConfig():
        camera = HikCameraThread(name, cam_config)
        camera.start()
        get_camera_provider = camera.get_latest_frame_getter
        camera_fps = camera.get_fps_getter()
        reader = None
    case RdrConfig():
        get_camera_provider = rdr_receiver.camera_service.get_latest_frame_getter
        camera_fps = rdr_receiver.camera_service.get_fps_getter()
        reader = None
    case _:
        raise ValueError(f"Unknown camera_service config: {cam_config}")

match cam_config.net_process:
    case True | str():
        # 暂时忽略从录制读取框框，当作完整的流程
        process = ProcessThread(cam_config.camera_id, cam_config.roi[2:4], get_camera_provider())
        get_process_provider = process.get_net_data_provider
        process_fps = process.get_fps_getter()
    case False:
        get_process_provider = lambda: lambda: None
        process_fps = lambda: 0.0
    case RdrReceive():
        get_process_provider = rdr_receiver.process.get_net_data_provider
        process_fps = rdr_receiver.process.get_fps_getter()
    case _:
        raise ValueError(f"Unknown net_process config: {cam_config.net_process}")
```

## 建议的解决方案

这种写法给录像读取用还算合理，因为录像必须要保证图像与网络的同步；但是对于实时的 `pyrdr` 完全可以做成异步的，能够减少代码的同时让抽象更有意义
将 `class ImageAndArmorClient` 改为 `class ArmorClient`，重复利用 `class ImageClient`，改变之后逻辑如下
```python
    match cam_config:
        case VideoConfig(path=path):
            reader = RecordReadManager(cam_config)
            get_camera_provider = reader.get_latest_frame_getter
            camera_fps = reader.get_fps_getter()
        case HikCameraConfig():
            camera = HikCameraThread(name, cam_config)
            camera.start()
            get_camera_provider = camera.get_latest_frame_getter
            camera_fps = camera.get_fps_getter()
            reader = None
        case RdrCameraConfig():
            camera = RdrCameraThread(cam_config)
            camera.start()
            get_camera_provider = camera.get_latest_frame_getter
            camera_fps = camera.get_fps_getter()
            reader = None
        case _:
            raise ValueError(f"Unknown camera_service config: {cam_config}")

    match cam_config.net_process:
        case True | str():
            # 暂时忽略从录制读取框框，当作完整的流程
            process = ProcessThread(cam_config.camera_id, cam_config.roi[2:4], get_camera_provider())
            get_process_provider = process.get_net_data_provider
            process_fps = process.get_fps_getter()
        case False:
            get_process_provider = lambda: lambda: None
            process_fps = lambda: 0.0
        case RdrReceive():
          process_receiver = RdrArmorReceiver(cam_config.net_process.endpoint)
            get_process_provider = rdr_receiver.process.get_net_data_provider
            process_fps = rdr_receiver.process.get_fps_getter()
        case _:
            raise ValueError(f"Unknown net_process config: {cam_config.net_process}")
```
干净又卫生

## 顺带一提

录像读取应该改一下了

```python
    match cam_config:
    case VideoConfig(path=path):
        reader = RecordReadManager(cam_config)
        reader.start()
        get_camera_provider = reader.get_latest_frame_getter
        camera_fps = reader.get_frame_fps_getter()
    case HikCameraConfig():
        camera = HikCameraThread(name, cam_config)
        camera.start()
        get_camera_provider = camera.get_latest_frame_getter
        camera_fps = camera.get_fps_getter()
    case RdrConfig():
        rdr_receiver = RdrReceiver(cam_config)
        rdr_receiver.start()
        get_camera_provider = rdr_receiver.camera_service.get_latest_frame_getter
        camera_fps = rdr_receiver.camera_service.get_fps_getter()
    case _:
        raise ValueError(f"Unknown camera_service config: {cam_config}")

match cam_config.net_process:
    case str():
        get_process_provider = reader.get_net_data_provider
        process_fps = reader.get_reader_fps_getter()
    case True | str():
        # 暂时忽略从录制读取框框，当作完整的流程
        process = ProcessThread(cam_config.camera_id, cam_config.roi[2:4], get_camera_provider())
        get_process_provider = process.get_net_data_provider
        process_fps = process.get_fps_getter()
    case False:
        get_process_provider = lambda: lambda: None
        process_fps = lambda: 0.0
    case RdrReceive():
        get_process_provider = rdr_receiver.process.get_net_data_provider
        process_fps = rdr_receiver.process.get_fps_getter()
    case _:
        raise ValueError(f"Unknown net_process config: {cam_config.net_process}")
```